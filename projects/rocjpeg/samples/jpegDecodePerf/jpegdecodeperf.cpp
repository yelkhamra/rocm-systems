/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "../rocjpeg_samples_utils.h"

struct DecodeInfo {
    std::vector<std::string> file_paths;
    RocJpegHandle rocjpeg_handle;         // used by producer thread for rocJpegGetImageInfo
    RocJpegHandle rocjpeg_decode_handle;  // used by consumer thread for rocJpegDecodeBatched
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles;
    uint64_t num_decoded_images;
    double images_per_sec;
    double image_size_in_mpixels_per_sec;
    uint64_t num_bad_jpegs;
    uint64_t num_jpegs_with_411_subsampling;
    uint64_t num_jpegs_with_unknown_subsampling;
    uint64_t num_jpegs_with_unsupported_resolution;
    int pipeline_depth;
};

// Thread-safe FIFO queue for pipelining the CPU producer (I/O + parse) with the GPU consumer (rocJpegDecodeBatched).
// Depth is naturally bounded by the number of BatchBuffer slots in the pipeline.
template<typename T>
struct BlockingQueue {
    void push(T item) {
        { std::lock_guard<std::mutex> lk(mtx_); q_.push(std::move(item)); }
        cv_.notify_one();
    }
    bool pop(T& item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&]{ return !q_.empty() || done_; });
        if (q_.empty()) return false;
        item = std::move(q_.front()); q_.pop(); return true;
    }
    void set_done() {
        { std::lock_guard<std::mutex> lk(mtx_); done_ = true; }
        cv_.notify_all();
    }
private:
    std::queue<T>           q_;
    std::mutex              mtx_;
    std::condition_variable cv_;
    bool                    done_ = false;
};

// Holds all per-batch state for one pipeline slot. Each buffer alternates between the producer
// (being filled with I/O + parsed stream data) and the consumer (being GPU-decoded).
struct BatchBuffer {
    int                                   handle_offset;            // start index into decode_info.rocjpeg_stream_handles
    std::vector<RocJpegStreamHandle>      stream_handles;           // compact handle array passed to rocJpegDecodeBatched
    std::vector<std::vector<char>>        batch_images;             // raw file bytes — kept alive until decode completes
    std::vector<RocJpegImage>             output_images;            // GPU output buffers
    std::vector<std::vector<uint32_t>>    allocated_channel_sizes;  // true GPU buffer capacities (high-water mark)
    std::vector<RocJpegDecodeParams>      decode_params;
    std::vector<RocJpegChromaSubsampling> subsamplings;
    std::vector<std::vector<uint32_t>>    widths;
    std::vector<std::vector<uint32_t>>    heights;
    std::vector<std::string>              base_file_names;
    int                                   current_batch_size = 0;
};

/**
 * @brief Decodes a batch of JPEG images and optionally saves the decoded images.
 *
 * @param decode_info parameters info for decoding a batch of jpeg images.
 * @param rocjpeg_utils Utility functions for RocJpeg operations.
 * @param decode_params Parameters for decoding the JPEG images (output_format, crop_rectangle)
 * @param save_images A boolean flag indicating whether to save the decoded images.
 * @param output_file_path The file path where the decoded images will be saved.
 * @param batch_size The number of images to be processed in each batch.
 */
void DecodeImages(DecodeInfo &decode_info, RocJpegUtils rocjpeg_utils, RocJpegDecodeParams &decode_params, bool save_images, std::string &output_file_path, int batch_size, int device_id) {

    uint8_t num_components;
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    std::string chroma_sub_sampling = "";
    uint32_t num_channels = 0;
    std::vector<uint32_t> temp_widths(ROCJPEG_MAX_COMPONENT, 0);
    std::vector<uint32_t> temp_heights(ROCJPEG_MAX_COMPONENT, 0);
    RocJpegChromaSubsampling temp_subsampling;
    std::string temp_base_file_name;
    int pipeline_depth = decode_info.pipeline_depth;

    CHECK_HIP(hipSetDevice(device_id));

    // Pre-allocate GPU memory at 2560x1440 to minimize hipMalloc/hipFree during decoding.
    // Sizes mirror GetChannelPitchAndSizes logic: only channels required by the selected
    // output_format are allocated. For NATIVE/YUV_PLANAR the subsampling is unknown upfront,
    // so worst-case sizes are used per channel (ch0: packed 4:2:2 = 2w; ch1/2: 4:4:4 full height).
    static constexpr uint32_t kMemAlignment = 16;
    static constexpr uint32_t kPreAllocWidth = 2560;
    static constexpr uint32_t kPreAllocHeight = 1440;
    auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
    const uint32_t aligned_w  = align_up(kPreAllocWidth, kMemAlignment);
    const uint32_t aligned_w2 = align_up(kPreAllocWidth, kMemAlignment) * 2; // packed YUV 4:2:2
    const uint32_t aligned_w3 = align_up(kPreAllocWidth, kMemAlignment) * 3; // packed RGB
    const uint32_t aligned_h  = align_up(kPreAllocHeight, kMemAlignment);
    uint32_t kPreAllocSizes[ROCJPEG_MAX_COMPONENT] = {};
    switch (decode_params.output_format) {
        case ROCJPEG_OUTPUT_Y:
            // 1 channel: luma plane only
            kPreAllocSizes[0] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_RGB:
            // 1 channel: packed RGB (pitch = width * 3)
            kPreAllocSizes[0] = aligned_w3 * aligned_h;
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            // 3 channels: planar R, G, B — all full resolution
            kPreAllocSizes[0] = kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            // up to 3 channels; worst case is 4:4:4 where all planes are full resolution
            kPreAllocSizes[0] = kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_NATIVE:
        default:
            // Subsampling unknown at pre-alloc time; cover worst case per channel:
            //   ch0: packed 4:2:2 uses pitch=2w (largest single-channel layout)
            //   ch1/ch2: 4:4:4 uses full height (largest multi-channel layout)
            kPreAllocSizes[0] = aligned_w2 * aligned_h;
            kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
    }

    // Initialize N pipeline buffers. Each buffer holds a full batch worth of GPU output memory,
    // stream handles, and CPU-side metadata. The producer fills one buffer while the GPU
    // decodes another, overlapping CPU and GPU work.
    //
    // GPU memory is pre-allocated only for the FIRST buffer (matching the original sequential
    // approach's footprint). All other buffers start with nullptr channel pointers and allocate
    // lazily on first use via the high-water mark grow path in fill_batch. This avoids
    // pipeline_depth × batch_size × channels upfront hipMalloc calls (e.g. 480 calls for
    // depth=5, batch=32) whose overhead and increased GPU memory footprint (cache pressure)
    // outweigh the pipeline benefit for I/O-bound workloads.
    std::vector<BatchBuffer> buffers(pipeline_depth);
    for (int b = 0; b < pipeline_depth; b++) {
        auto& buf = buffers[b];
        buf.handle_offset = b * batch_size;
        buf.batch_images.resize(batch_size);
        buf.stream_handles.resize(batch_size);
        buf.output_images.resize(batch_size);
        buf.allocated_channel_sizes.assign(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        buf.decode_params.assign(batch_size, decode_params);
        buf.subsamplings.resize(batch_size);
        buf.widths.assign(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        buf.heights.assign(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        buf.base_file_names.resize(batch_size);
        if (b == 0) {
            // Pre-allocate GPU memory at steady-state size for the first buffer only.
            for (int slot = 0; slot < batch_size; slot++) {
                for (int n = 0; n < ROCJPEG_MAX_COMPONENT; n++) {
                    if (kPreAllocSizes[n] > 0) {
                        CHECK_HIP(hipMalloc(&buf.output_images[slot].channel[n], kPreAllocSizes[n]));
                        buf.allocated_channel_sizes[slot][n] = kPreAllocSizes[n];
                    }
                }
            }
        }
        // Buffers b > 0: channel pointers remain nullptr; the grow check in fill_batch
        // allocates them on first use and the high-water mark prevents re-allocation thereafter.
    }

    // RAII guard: frees all GPU buffers across all pipeline slots on any exit path.
    struct BuffersGuard {
        std::vector<BatchBuffer>& bufs;
        ~BuffersGuard() {
            for (auto& buf : bufs) {
                for (auto& img : buf.output_images) {
                    for (int n = 0; n < ROCJPEG_MAX_COMPONENT; n++) {
                        if (img.channel[n] != nullptr) {
                            (void)hipFree((void*)img.channel[n]);
                            img.channel[n] = nullptr;
                        }
                    }
                }
            }
        }
    } buffers_guard{buffers};

    // Seed the free queue with all pipeline buffer pointers.
    BlockingQueue<BatchBuffer*> free_queue, ready_queue;
    for (auto& buf : buffers) free_queue.push(&buf);

    // Consumer thread stats — written only by the consumer, read after join().
    uint64_t consumer_num_decoded    = 0;
    double   consumer_decode_time_ms = 0;
    double   consumer_mpixels_all    = 0;

    // Lambda: parse files [file_start, file_end) into buf. Returns false on fatal I/O error.
    // Used for both the warmup batch (synchronous, before the consumer thread) and all
    // subsequent batches in the producer loop.
    auto fill_batch = [&](BatchBuffer* buf, int file_start, int file_end) -> bool {
        buf->current_batch_size = 0;
        for (int j = file_start; j < file_end; j++) {
            int index = j - file_start;
            temp_base_file_name = decode_info.file_paths[j].substr(decode_info.file_paths[j].find_last_of("/\\") + 1);
            std::ifstream input(decode_info.file_paths[j].c_str(), std::ios::in | std::ios::binary | std::ios::ate);
            if (!input.is_open()) {
                std::cerr << "ERROR: Cannot open image: " << decode_info.file_paths[j] << std::endl;
                return false;
            }
            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            if (static_cast<std::streamsize>(buf->batch_images[index].size()) < file_size)
                buf->batch_images[index].resize(file_size);
            if (!input.read(buf->batch_images[index].data(), file_size)) {
                std::cerr << "ERROR: Cannot read from file: " << decode_info.file_paths[j] << std::endl;
                return false;
            }
            RocJpegStreamHandle stream_handle = decode_info.rocjpeg_stream_handles[buf->handle_offset + index];
            RocJpegStatus rocjpeg_status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(buf->batch_images[index].data()), file_size, stream_handle);
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                decode_info.num_bad_jpegs++;
                continue;
            }
            CHECK_ROCJPEG(rocJpegGetImageInfo(decode_info.rocjpeg_handle, stream_handle, &num_components, &temp_subsampling, temp_widths.data(), temp_heights.data()));
            rocjpeg_utils.GetChromaSubsamplingStr(temp_subsampling, chroma_sub_sampling);
            if (temp_widths[0] < 64 || temp_heights[0] < 64) {
                decode_info.num_jpegs_with_unsupported_resolution++;
                continue;
            }
            if (temp_subsampling == ROCJPEG_CSS_411 || temp_subsampling == ROCJPEG_CSS_UNKNOWN) {
                if (temp_subsampling == ROCJPEG_CSS_411)    decode_info.num_jpegs_with_411_subsampling++;
                if (temp_subsampling == ROCJPEG_CSS_UNKNOWN) decode_info.num_jpegs_with_unknown_subsampling++;
                continue;
            }
            int slot = buf->current_batch_size;
            if (rocjpeg_utils.GetChannelPitchAndSizes(buf->decode_params[slot], temp_subsampling, temp_widths.data(), temp_heights.data(), num_channels, buf->output_images[slot], channel_sizes)) {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return false;
            }
            for (int n = 0; n < static_cast<int>(num_channels); n++) {
                if (channel_sizes[n] > buf->allocated_channel_sizes[slot][n]) {
                    if (buf->output_images[slot].channel[n] != nullptr) {
                        CHECK_HIP(hipFree((void*)buf->output_images[slot].channel[n]));
                        buf->output_images[slot].channel[n] = nullptr;
                    }
                    CHECK_HIP(hipMalloc(&buf->output_images[slot].channel[n], channel_sizes[n]));
                    buf->allocated_channel_sizes[slot][n] = channel_sizes[n];
                }
            }
            buf->stream_handles[slot]  = stream_handle;
            buf->subsamplings[slot]    = temp_subsampling;
            buf->widths[slot]          = temp_widths;
            buf->heights[slot]         = temp_heights;
            buf->base_file_names[slot] = temp_base_file_name;
            buf->current_batch_size++;
        }
        return true;
    };

    // Lambda: save decoded images from buf (mirrors the consumer's save block).
    auto save_batch = [&](BatchBuffer* buf) {
        bool is_roi_valid = false;
        uint32_t roi_width, roi_height;
        for (int b = 0; b < buf->current_batch_size; b++) {
            std::string image_save_path = output_file_path;
            roi_width  = buf->decode_params[b].crop_rectangle.right  - buf->decode_params[b].crop_rectangle.left;
            roi_height = buf->decode_params[b].crop_rectangle.bottom - buf->decode_params[b].crop_rectangle.top;
            is_roi_valid = (roi_width > 0 && roi_height > 0 && roi_width <= buf->widths[b][0] && roi_height <= buf->heights[b][0]);
            uint32_t width  = is_roi_valid ? roi_width  : buf->widths[b][0];
            uint32_t height = is_roi_valid ? roi_height : buf->heights[b][0];
            rocjpeg_utils.GetOutputFileExt(decode_params.output_format, buf->base_file_names[b], width, height, buf->subsamplings[b], image_save_path);
            rocjpeg_utils.SaveImage(image_save_path, &buf->output_images[b], width, height, buf->subsamplings[b], decode_params.output_format);
        }
    };

    // Warmup: parse and decode the first batch synchronously before spawning the consumer
    // thread. The first rocJpegDecodeBatched call incurs a one-time VA-API GPU pipeline
    // initialization cost (vaBeginPicture context creation, ~29ms in traces) that would
    // otherwise stall the producer — which fills a batch in ~2ms — for the entire
    // initialization window. After this synchronous call the VA context is live and all
    // subsequent pipeline decodes run at steady-state latency (~9ms).
    int pipeline_file_start = 0;
    if (!decode_info.file_paths.empty()) {
        BatchBuffer* buf = nullptr;
        free_queue.pop(buf);
        int warmup_end = std::min(batch_size, static_cast<int>(decode_info.file_paths.size()));
        if (!fill_batch(buf, 0, warmup_end)) {
            free_queue.push(buf);
            return;
        }
        if (buf->current_batch_size > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_ROCJPEG(rocJpegDecodeBatched(
                decode_info.rocjpeg_decode_handle,
                buf->stream_handles.data(),
                buf->current_batch_size,
                buf->decode_params.data(),
                buf->output_images.data()));
            auto t1 = std::chrono::high_resolution_clock::now();
            consumer_decode_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            for (int b = 0; b < buf->current_batch_size; b++)
                consumer_mpixels_all += static_cast<double>(buf->widths[b][0]) * static_cast<double>(buf->heights[b][0]) / 1000000.0;
            consumer_num_decoded += buf->current_batch_size;
            decode_info.num_decoded_images += buf->current_batch_size;
            if (save_images) save_batch(buf);
        }
        free_queue.push(buf);
        pipeline_file_start = warmup_end;
    }

    // Consumer thread: dequeues filled buffers, runs rocJpegDecodeBatched at steady-state
    // latency (VA context already initialized by the warmup above), accumulates timing stats,
    // optionally saves images, then returns the buffer to the free queue.
    std::thread decoder_thread([&]() {
        BatchBuffer* buf = nullptr;
        while (ready_queue.pop(buf)) {
            if (buf->current_batch_size > 0) {
                auto t0 = std::chrono::high_resolution_clock::now();
                CHECK_ROCJPEG(rocJpegDecodeBatched(
                    decode_info.rocjpeg_decode_handle,
                    buf->stream_handles.data(),
                    buf->current_batch_size,
                    buf->decode_params.data(),
                    buf->output_images.data()));
                auto t1 = std::chrono::high_resolution_clock::now();
                consumer_decode_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                for (int b = 0; b < buf->current_batch_size; b++)
                    consumer_mpixels_all += static_cast<double>(buf->widths[b][0]) * static_cast<double>(buf->heights[b][0]) / 1000000.0;
                consumer_num_decoded += buf->current_batch_size;
                if (save_images) save_batch(buf);
            }
            free_queue.push(buf);
        }
    });

    // Producer loop: fills pipeline buffers starting after the warmup batch and hands them
    // to the consumer thread via ready_queue.
    for (int i = pipeline_file_start; i < static_cast<int>(decode_info.file_paths.size()); i += batch_size) {
        BatchBuffer* buf = nullptr;
        free_queue.pop(buf);
        int batch_end = std::min(i + batch_size, static_cast<int>(decode_info.file_paths.size()));
        if (!fill_batch(buf, i, batch_end)) {
            ready_queue.set_done();
            decoder_thread.join();
            return;
        }
        decode_info.num_decoded_images += buf->current_batch_size;
        ready_queue.push(buf);
    }

    ready_queue.set_done();
    decoder_thread.join();

    double avg_time_per_image = consumer_num_decoded > 0 ? consumer_decode_time_ms / consumer_num_decoded : 0;
    decode_info.images_per_sec = avg_time_per_image > 0 ? 1000.0 / avg_time_per_image : 0;
    decode_info.image_size_in_mpixels_per_sec = consumer_num_decoded > 0 ? decode_info.images_per_sec * consumer_mpixels_all / consumer_num_decoded : 0;

}

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    int num_threads = 1;
    int batch_size = 1;
    int pipeline_depth = 2;
    bool is_dir = false;
    bool is_file = false;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    std::vector<DecodeInfo> decode_info_per_thread;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, &num_threads, &batch_size, argc, argv, &pipeline_depth);
    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    if (num_threads > file_paths.size()) {
        num_threads = file_paths.size();
    }

    decode_info_per_thread.resize(num_threads);

    for (int i = 0; i < num_threads; i++) {
        CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &decode_info_per_thread[i].rocjpeg_handle));
        CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &decode_info_per_thread[i].rocjpeg_decode_handle));
        int total_stream_handles = pipeline_depth * batch_size;
        decode_info_per_thread[i].rocjpeg_stream_handles.resize(total_stream_handles);
        for (auto j = 0; j < total_stream_handles; j++) {
            CHECK_ROCJPEG(rocJpegStreamCreate(&decode_info_per_thread[i].rocjpeg_stream_handles[j]));
        }
        decode_info_per_thread[i].num_decoded_images = 0;
        decode_info_per_thread[i].images_per_sec = 0;
        decode_info_per_thread[i].image_size_in_mpixels_per_sec = 0;
        decode_info_per_thread[i].num_bad_jpegs = 0;
        decode_info_per_thread[i].num_jpegs_with_411_subsampling = 0;
        decode_info_per_thread[i].num_jpegs_with_unknown_subsampling = 0;
        decode_info_per_thread[i].num_jpegs_with_unsupported_resolution = 0;
        decode_info_per_thread[i].pipeline_depth = pipeline_depth;
    }

    ThreadPool thread_pool(num_threads);

    size_t files_per_thread = file_paths.size() / num_threads;
    size_t remaining_files = file_paths.size() % num_threads;
    size_t start_index = 0;
    for (int i = 0; i < num_threads; i++) {
        size_t end_index = start_index + files_per_thread + (i < remaining_files ? 1 : 0);
        decode_info_per_thread[i].file_paths.assign(file_paths.begin() + start_index, file_paths.begin() + end_index);
        start_index = end_index;
    }

    std::cout << "Decoding started with " << num_threads << " threads, please wait!" << std::endl;
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        thread_pool.ExecuteJob(std::bind(DecodeImages, std::ref(decode_info_per_thread[i]), rocjpeg_utils, std::ref(decode_params), save_images, std::ref(output_file_path), batch_size, device_id));
    }
    thread_pool.JoinThreads();
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    double total_wall_time_in_sec = std::chrono::duration<double>(overall_end_time - overall_start_time).count();

    uint64_t total_decoded_images = 0;
    double total_images_per_sec = 0;
    double total_image_size_in_mpixels_per_sec = 0;
    uint64_t total_num_bad_jpegs = 0;
    uint64_t total_num_jpegs_with_411_subsampling = 0;
    uint64_t total_num_jpegs_with_unknown_subsampling = 0;
    uint64_t total_num_jpegs_with_unsupported_resolution = 0;

    for (auto i = 0; i < num_threads; i++) {
        total_decoded_images += decode_info_per_thread[i].num_decoded_images;
        total_image_size_in_mpixels_per_sec += decode_info_per_thread[i].image_size_in_mpixels_per_sec;
        total_images_per_sec += decode_info_per_thread[i].images_per_sec;
        total_num_bad_jpegs += decode_info_per_thread[i].num_bad_jpegs;
        total_num_jpegs_with_411_subsampling += decode_info_per_thread[i].num_jpegs_with_411_subsampling;
        total_num_jpegs_with_unknown_subsampling += decode_info_per_thread[i].num_jpegs_with_unknown_subsampling;
        total_num_jpegs_with_unsupported_resolution += decode_info_per_thread[i].num_jpegs_with_unsupported_resolution;
    }

    std::cout << "Total decoded images: " << total_decoded_images << std::endl;
    if (total_num_bad_jpegs || total_num_jpegs_with_411_subsampling || total_num_jpegs_with_unknown_subsampling || total_num_jpegs_with_unsupported_resolution) {
        std::cout << "Total skipped images: " << total_num_bad_jpegs + total_num_jpegs_with_411_subsampling + total_num_jpegs_with_unknown_subsampling + total_num_jpegs_with_unsupported_resolution;
        if (total_num_bad_jpegs) {
            std::cout << " ,total images that cannot be parsed: " << total_num_bad_jpegs;
        }
        if (total_num_jpegs_with_411_subsampling) {
            std::cout << " ,total images with YUV 4:1:1 chroma subsampling: " << total_num_jpegs_with_411_subsampling;
        }
        if (total_num_jpegs_with_unknown_subsampling) {
            std::cout << " ,total images with unknown chroma subsampling: " << total_num_jpegs_with_unknown_subsampling;
        }
        if (total_num_jpegs_with_unsupported_resolution) {
            std::cout << " ,total images with unsupported_resolution: " << total_num_jpegs_with_unsupported_resolution;
        }
        std::cout << std::endl;
    }

    if (total_decoded_images > 0) {
        std::cout << "Average processing time per image (ms): " << 1000 / total_images_per_sec << std::endl;
        std::cout << "Average decoded images per sec (Images/Sec): " << total_images_per_sec << std::endl;
        std::cout << "Average decoded images size (Mpixels/Sec): " << total_image_size_in_mpixels_per_sec << std::endl;
    }

    if (total_wall_time_in_sec >= 3600) {
        std::cout << "Total wall time (hours): " << total_wall_time_in_sec / 3600 << std::endl;
    } else if (total_wall_time_in_sec >= 60) {
        std::cout << "Total wall time (min): " << total_wall_time_in_sec / 60 << std::endl;
    } else {
        std::cout << "Total wall time (sec): " << total_wall_time_in_sec << std::endl;
    }

    for (int i = 0; i < num_threads; i++) {
        CHECK_ROCJPEG(rocJpegDestroy(decode_info_per_thread[i].rocjpeg_handle));
        CHECK_ROCJPEG(rocJpegDestroy(decode_info_per_thread[i].rocjpeg_decode_handle));
        for (auto& h : decode_info_per_thread[i].rocjpeg_stream_handles) {
            CHECK_ROCJPEG(rocJpegStreamDestroy(h));
        }
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
