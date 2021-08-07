#define _GNU_SOURCE
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define KiB ((size_t)1024)
#define MiB (KiB * KiB)
#define GiB (MiB * KiB)

#define MI_INTPTR_SHIFT (3)
#define MI_SEGMENT_SLICE_SHIFT (13 + MI_INTPTR_SHIFT) // 64kb
#define MI_SEGMENT_SHIFT (7 + MI_SEGMENT_SLICE_SHIFT) //  8mb
#define MI_SEGMENT_SIZE (1ULL << MI_SEGMENT_SHIFT)
#define MI_SEGMENT_SLICE_SIZE (1ULL << MI_SEGMENT_SLICE_SHIFT)
#define MI_SLICES_PER_SEGMENT                                                  \
    128 // (MI_SEGMENT_SIZE / MI_SEGMENT_SLICE_SIZE) // 1024

#define KK_HINT_BASE ((uintptr_t)2 << 40) // 2TiB start
#define KK_HINT_AREA                                                           \
    ((uintptr_t)4 << 40) // upto 6TiB   (since before win8 there is "only" 8TiB
                         // available to processes)
#define KK_HINT_MAX                                                            \
    ((uintptr_t)30                                                             \
     << 40) // wrap after 30TiB (area after 32TiB is used for huge OS pages)

#define MI_CACHE_LINE 64

#define MI_SEGMENT_MASK ((uintptr_t)MI_SEGMENT_SIZE - 1)

// Maximum number of size classes. (spaced exponentially in 12.5% increments)
#define MI_BIN_HUGE (73U)
#define MI_BIN_FULL (MI_BIN_HUGE + 1)

#define MI_SMALL_PAGE_SHIFT (MI_SEGMENT_SLICE_SHIFT)   // 64kb
#define MI_MEDIUM_PAGE_SHIFT (3 + MI_SMALL_PAGE_SHIFT) // 512kb
#define MI_SMALL_PAGE_SIZE (1ULL << MI_SMALL_PAGE_SHIFT)
#define MI_MEDIUM_PAGE_SIZE (1ULL << MI_MEDIUM_PAGE_SHIFT)
#define MI_MEDIUM_OBJ_SIZE_MAX (MI_MEDIUM_PAGE_SIZE / 4) // 128kb on 64-bit
#define MI_MEDIUM_OBJ_WSIZE_MAX                                                \
    (MI_MEDIUM_OBJ_SIZE_MAX / MI_INTPTR_SIZE) // 64kb on 64-bit

#define MI_INTPTR_SIZE (1 << MI_INTPTR_SHIFT)
#define MI_INTPTR_BITS (MI_INTPTR_SIZE * 8)

// Empty page queues for every bin
#define QNULL(sz)                                                              \
    { NULL, NULL, (sz) * sizeof(uintptr_t) }
#define MI_PAGE_QUEUES_EMPTY                                                   \
    {                                                                          \
        QNULL(1), QNULL(1), QNULL(2), QNULL(3), QNULL(4), QNULL(5), QNULL(6),  \
            QNULL(7), QNULL(8), /* 8 */                                        \
            QNULL(10), QNULL(12), QNULL(14), QNULL(16), QNULL(20), QNULL(24),  \
            QNULL(28), QNULL(32), /* 16 */                                     \
            QNULL(40), QNULL(48), QNULL(56), QNULL(64), QNULL(80), QNULL(96),  \
            QNULL(112), QNULL(128), /* 24 */                                   \
            QNULL(160), QNULL(192), QNULL(224), QNULL(256), QNULL(320),        \
            QNULL(384), QNULL(448), QNULL(512), /* 32 */                       \
            QNULL(640), QNULL(768), QNULL(896), QNULL(1024), QNULL(1280),      \
            QNULL(1536), QNULL(1792), QNULL(2048), /* 40 */                    \
            QNULL(2560), QNULL(3072), QNULL(3584), QNULL(4096), QNULL(5120),   \
            QNULL(6144), QNULL(7168), QNULL(8192), /* 48 */                    \
            QNULL(10240), QNULL(12288), QNULL(14336), QNULL(16384),            \
            QNULL(20480), QNULL(24576), QNULL(28672), QNULL(32768), /* 56 */   \
            QNULL(40960), QNULL(49152), QNULL(57344), QNULL(65536),            \
            QNULL(81920), QNULL(98304), QNULL(114688), QNULL(131072), /* 64 */ \
            QNULL(163840), QNULL(196608), QNULL(229376), QNULL(262144),        \
            QNULL(327680), QNULL(393216), QNULL(458752),                       \
            QNULL(524288), /* 72 */                                            \
            QNULL(MI_MEDIUM_OBJ_WSIZE_MAX + 1 /* 655360, Huge queue */),       \
            QNULL(MI_MEDIUM_OBJ_WSIZE_MAX + 2) /* Full queue */                \
    }

#define SEGMENT_SIZE 128 * MI_SEGMENT_SLICE_SIZE

typedef struct mi_block_s {
    struct mi_block_s *next;
} mi_block_t;

typedef struct mi_page_s {
    int idx;
    uint32_t block_size;
    mi_block_t *free;
    size_t reserved;
    size_t used;
    struct mi_page_s *prev;
    struct mi_page_s *next;
    size_t slice_count;
    char dummy[32]; // used dummy to correct alignment
} mi_page_t;

typedef struct mi_segment_s {
    void *addr;
    size_t size;
    struct mi_segment_s *next;
    struct mi_segment_s *prev;
    char dummy[96]; // used dummy to correct alignment
    mi_page_t slices[MI_SLICES_PER_SEGMENT];
} mi_segment_t;

typedef struct mi_page_queue_s {
    mi_page_t *first;
    mi_page_t *last;
    size_t block_size;
} mi_page_queue_t;

#define MI_SEGMENT_BIN_MAX (35) // 35 == mi_segment_bin(MI_SLICES_PER_SEGMENT

// A "span" is is an available range of slices. The span queues keep
// track of slice spans of at most the given `slice_count` (but more than the
// previous size class).
typedef struct mi_span_queue_s {
    mi_page_t *first;
    mi_page_t *last;
    size_t slice_count;
} mi_span_queue_t;

// Segments thread local data
typedef struct mi_segments_tld_s {
    mi_span_queue_t spans;  // free slice spans inside segments
    size_t count;           // current number of segments;
    mi_segment_t *segments; // store segments in order to free
} mi_segments_tld_t;

static __attribute__((aligned(MI_CACHE_LINE))) uintptr_t aligned_base;
static mi_page_queue_t page_queues[MI_BIN_FULL + 1] =
    MI_PAGE_QUEUES_EMPTY; // queue of pages for each size class (or "bin")

static mi_segments_tld_t cld = { { NULL, NULL, 128 }, 0 };

static void mi_span_queue_push(mi_span_queue_t *sq, mi_page_t *slice);
static void *alloc_memory_from_os();

/* -----------------------------------------------------------
   helper
----------------------------------------------------------- */

/* base-2 integer ceiling */
static unsigned int iceil2(unsigned int x) {
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x + 1;
}

// "bit scan reverse": Return index of the highest bit (or MI_INTPTR_BITS if
// `x` is zero)
static inline size_t mi_bsr(uintptr_t x) {
    return (x == 0 ? MI_INTPTR_BITS : MI_INTPTR_BITS - 1 - __builtin_clzl(x));
}

// Align a byte size to a size in _machine words_,
// i.e. byte size == `wsize*sizeof(void*)`.
static inline size_t _mi_wsize_from_size(size_t size) {
    return (size + sizeof(uintptr_t) - 1) / sizeof(uintptr_t);
}

// Return the bin for a given field size.
// Returns MI_BIN_HUGE if the size is too large.
// We use `wsize` for the size in "machine word sizes",
// i.e. byte size == `wsize*sizeof(void*)`.
extern inline uint8_t _mi_bin(size_t size) {
    size_t wsize = _mi_wsize_from_size(size);
    uint8_t bin;
    if (wsize <= 1) {
        bin = 1;
    }
#if defined(MI_ALIGN4W)
    else if (wsize <= 4) {
        bin = (uint8_t)((wsize + 1) & ~1); // round to double word sizes
    }
#elif defined(MI_ALIGN2W)
    else if (wsize <= 8) {
        bin = (uint8_t)((wsize + 1) & ~1); // round to double word sizes
    }
#else
    else if (wsize <= 8) {
        bin = (uint8_t)wsize;
    }
#endif
    else if (wsize > MI_MEDIUM_OBJ_WSIZE_MAX) {
        bin = MI_BIN_HUGE;
    } else {
#if defined(MI_ALIGN4W)
        if (wsize <= 16) {
            wsize = (wsize + 3) & ~3;
        } // round to 4x word sizes
#endif
        wsize--;
        // find the highest bit
        uint8_t b = (uint8_t)mi_bsr(wsize); // note: wsize != 0
        // and use the top 3 bits to determine the bin (~12.5% worst internal
        // fragmentation).
        // - adjust with 3 because we use do not round the first 8 sizes
        //   which each get an exact bin
        bin = ((b << 2) + (uint8_t)((wsize >> (b - 2)) & 0x03)) - 3;
    }
    return bin;
}

/* -----------------------------------------------------------
   Segment related function
----------------------------------------------------------- */

static uint8_t *_mi_segment_page_start_from_slice(const mi_segment_t *segment,
                                                  const mi_page_t *page,
                                                  size_t *page_size) {
    ptrdiff_t idx = page - segment->slices;
    size_t psize = MI_SEGMENT_SLICE_SIZE;
    if (page_size != NULL)
        *page_size = psize;
    return (uint8_t *)segment + (idx * MI_SEGMENT_SLICE_SIZE);
}

// Start of the page available memory; can be used on uninitialized pages
uint8_t *_mi_segment_page_start(const mi_segment_t *segment,
                                const mi_page_t *page, size_t *page_size) {
    return (uint8_t *)_mi_segment_page_start_from_slice(segment, page, page_size);
}

mi_segment_t *_mi_ptr_segment(mi_page_t *slice) {
    return (mi_segment_t *)((uintptr_t)slice & ~MI_SEGMENT_MASK);
}

static void mi_segment_span_free(mi_segment_t *segment, size_t slice_index,
                                 size_t slice_count, mi_segments_tld_t *tld) {
    mi_span_queue_t *sq = &tld->spans;

    // set first and last slice (the intermediates can be undetermined)
    mi_page_t *slice = &segment->slices[slice_index];
    slice->slice_count = (uint32_t)slice_count;

    // and push it on the free page queue (if it was not a huge page)
    if (sq != NULL)
        mi_span_queue_push(sq, slice);
}

static void mi_segment_init() {
    mi_segment_t *p = (mi_segment_t *)alloc_memory_from_os();
    cld.count++;
    mi_segment_span_free(p, 1, 127, &cld);
    assert(p != MAP_FAILED && p != NULL);
    if (cld.segments == NULL) {
        cld.segments = p;
    } else {
        p->next = cld.segments;
        cld.segments->prev = p;
        cld.segments = p;
    }
}

static inline size_t mi_slice_index(mi_page_t *slice) {
    mi_segment_t *segment = _mi_ptr_segment(slice);
    ptrdiff_t index = slice - segment->slices;
    return index;
}

static void mi_segment_slice_split(mi_segment_t *segment, mi_page_t *slice,
                                   size_t slice_count, mi_segments_tld_t *tld) {
    if (slice->slice_count <= slice_count)
        return;
    size_t next_index = mi_slice_index(slice) + slice_count;
    size_t next_count = slice->slice_count - slice_count;
    mi_segment_span_free(segment, next_index, next_count, tld);
    slice->slice_count = (uint32_t)slice_count;
}

/* -----------------------------------------------------------
  page related function
----------------------------------------------------------- */

static inline mi_block_t *mi_page_block_at(const mi_page_t *page,
                                           void *page_start, size_t block_size,
                                           size_t i) {
    return (mi_block_t *)((uint8_t *)page_start + (i * block_size));
}

static void mi_page_list_extend(mi_page_t *page, size_t reversed,
                                void *page_area, size_t block_size) {
    mi_block_t *const start = mi_page_block_at(page, page_area, block_size, 0);

    // initialize a sequential free list
    mi_block_t *const last =
        mi_page_block_at(page, page_area, block_size, 0 + page->reserved - 1);
    mi_block_t *block = start;
    while (block <= last) {
        mi_block_t *next = (mi_block_t *)((uint8_t *)block + block_size);
        block->next = next;
        block = next;
    }
    last->next = page->free;
    page->free = start;
}

/* -----------------------------------------------------------
  span related function
----------------------------------------------------------- */

static void mi_span_queue_push(mi_span_queue_t *sq, mi_page_t *slice) {
    // todo: or push to the end?
    slice->prev = NULL; // paranoia
    slice->next = sq->first;
    sq->first = slice;
    if (slice->next != NULL)
        slice->next->prev = slice;
    else
        sq->last = slice;
}

static void mi_span_queue_delete(mi_span_queue_t *sq, mi_page_t *slice) {
    // should work too if the queue does not contain slice (which can happen
    // during reclaim)
    if (slice->prev != NULL)
        slice->prev->next = slice->next;
    if (slice == sq->first)
        sq->first = slice->next;
    if (slice->next != NULL)
        slice->next->prev = slice->prev;
    if (slice == sq->last)
        sq->last = slice->prev;
    slice->prev = NULL;
    slice->next = NULL;
}

/* -----------------------------------------------------------
  Queue of pages with free blocks
----------------------------------------------------------- */

static inline mi_page_queue_t *mi_page_queue(size_t size) {
    return &page_queues[_mi_bin(size)];
}

static mi_page_queue_t *mi_page_queue_of(const mi_page_t *page) {
    uint8_t bin = _mi_bin(page->block_size);
    mi_page_queue_t *pq = &page_queues[bin];
    return pq;
}

static void mi_page_queue_push(mi_page_queue_t *queue, mi_page_t *page) {
    page->next = queue->first;
    page->prev = NULL;
    if (queue->first != NULL) {
        queue->first->prev = page;
        queue->first = page;
    } else {
        queue->first = queue->last = page;
    }
}

static void mi_page_queue_remove(mi_page_queue_t *queue, mi_page_t *page) {
    if (page->prev != NULL)
        page->prev->next = page->next;
    if (page->next != NULL)
        page->next->prev = page->prev;
    if (page == queue->last)
        queue->last = page->prev;
    if (page == queue->first)
        queue->first = page->next;
    page->next = NULL;
    page->prev = NULL;
}

/* -----------------------------------------------------------
  OS API: alloc, free
----------------------------------------------------------- */

static void *mi_os_get_aligned_hint(size_t try_alignment, size_t size) {
    if (try_alignment == 0 || try_alignment > MI_SEGMENT_SIZE)
        return NULL;
    if ((size % MI_SEGMENT_SIZE) != 0)
        return NULL;
    if (size > 1 * GiB)
        return NULL; // guarantee the chance of fixed valid address is at most
                     // 1/(KK_HINT_AREA / 1<<30) = 1/4096.
    uintptr_t hint = aligned_base;
    aligned_base += size;
    if (hint == 0 || hint > KK_HINT_MAX) { // wrap or initialize
        uintptr_t init = KK_HINT_BASE;
        aligned_base = init;
        hint = aligned_base;
        aligned_base += size; // this may still give 0 or > KK_HINT_MAX but that
                              // is ok, it is a hint after all
    }
    if (hint % try_alignment != 0)
        return NULL;
    return (void *)hint;
}

static void *alloc_memory_from_os() {
    void *hint, *p;
    int flags, protect_flags;
    size_t try_alignment, size;

    flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    protect_flags = PROT_WRITE | PROT_READ;
    try_alignment = size = SEGMENT_SIZE;

    if ((hint = mi_os_get_aligned_hint(try_alignment, size)) != NULL) {
        p = mmap(hint, size, protect_flags, flags, -1, 0);
    }
    return p;
}

static void mi_segment_free() {
    mi_segment_t *segment = cld.segments;
    while (segment) {
        mi_segment_t *next = segment->next;
        munmap(segment, SEGMENT_SIZE);
        segment = next;
    }
}




void *mi_malloc(size_t size) {
    mi_block_t *block = NULL;
    mi_page_t *page = NULL;
    size_t page_size;
    void *start;

    size = iceil2(size);
    if (size < 8) // size need larger than pointer size
        size = 8;
    mi_page_queue_t *pq = mi_page_queue(size);
    page = pq->first;
    // fast path
    while (page) {
        if (page->free) {
            block = page->free;
            break;
        }
        page = page->next;
    }
    // slow path
    if (block == NULL) {
        mi_span_queue_t *sq = &cld.spans;
        for (mi_page_t *slice = sq->first; slice != NULL; slice = slice->next) {
            if (slice->slice_count >= 1) {
                // found one
                mi_span_queue_delete(sq, slice);
                mi_segment_t *segment = _mi_ptr_segment(slice);
                if (slice->slice_count > 1) {
                    mi_segment_slice_split(segment, slice, 1, &cld);
                }
                page = slice;

                start = _mi_segment_page_start(segment, page, &page_size);
                page->block_size = size;
                page->reserved = page_size / page->block_size;
                mi_page_list_extend(page, page->reserved, start,
                                    page->block_size);
                break;
            }
        }
        if (!page) {
            // out of memory, allocate a new segment and try again
            mi_segment_init();
            return mi_malloc(size);
        }

        mi_page_queue_push(pq, page);
        block = page->free;
    }

    assert(block != NULL);
    page->used++;
    page->free = block->next;
    assert(page->used <= page->reserved);

    return block;
}

void mi_free(void *p) {
    mi_segment_t *segment = (mi_segment_t *)((uintptr_t)p & ~MI_SEGMENT_MASK);

    ptrdiff_t diff = (uint8_t *)p - (uint8_t *)segment;
    uintptr_t idx = (uintptr_t)diff >> MI_SEGMENT_SLICE_SHIFT;
    mi_page_t *page = (mi_page_t *)&segment->slices[idx];

    ((mi_block_t *)p)->next = page->free;
    page->free = p;
    page->used--;

    if (page->used == 0) {
        mi_page_queue_t *pq = mi_page_queue_of(page);
        mi_page_queue_remove(pq, page);
        mi_segment_span_free(segment, mi_slice_index(page), 1, &cld);
        page->free = NULL;
    }
}

int main(void) {
    mi_segment_init();

    // test code
    for (int i = 0; i < 5000000; i++) {
        int sz = random() % 64;
        /* also alloc some larger chunks  */
        if (random() % 100 == 0)
            sz = random() % 10000;
        if (!sz)
            sz = 1; /* avoid zero allocations */
        int *ip = (int *)mi_malloc(sz);
        *ip = 7;

        /* randomly repool some of them */
        if (random() % 10 == 0) /* repool, known size */
            mi_free(ip);
        if (i % 10000 == 0 && i > 0) {
            putchar('.');
            if (i % 700000 == 0)
                putchar('\n');
        }
    }

    mi_segment_free();
    return 0;
}