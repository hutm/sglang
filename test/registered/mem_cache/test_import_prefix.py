"""Adopting KV that already sits in specific pages as a cached prefix."""

import unittest
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

PAGE = 4


def _paged(num_pages: int = 8):
    return PagedTokenToKVPoolAllocator(
        size=num_pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=SimpleNamespace(),
        need_sort=True,
    )


def _cache(allocator, tree_core_backend=None):
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=allocator,
            page_size=allocator.page_size,
            tree_components=(ComponentType.FULL,),
            tree_core_backend=tree_core_backend,
        )
    )


def test_alloc_pages_claims_exactly_the_requested_pages():
    allocator = _paged()
    before = allocator.available_size()
    indices = allocator.alloc_pages(torch.tensor([5, 2]))
    assert indices.tolist() == [20, 21, 22, 23, 8, 9, 10, 11]
    assert allocator.available_size() == before - 2 * PAGE
    remaining = set(allocator.get_all_free_pages().tolist())
    assert 5 not in remaining and 2 not in remaining


def test_alloc_pages_is_all_or_nothing():
    allocator = _paged()
    allocator.alloc_pages(torch.tensor([3]))
    before = allocator.get_all_free_pages().clone()
    # Page 3 is taken, page 0 is the padding page, and duplicates are invalid.
    assert allocator.alloc_pages(torch.tensor([1, 3])) is None
    assert allocator.alloc_pages(torch.tensor([0])) is None
    assert allocator.alloc_pages(torch.tensor([4, 4])) is None
    assert torch.equal(allocator.get_all_free_pages(), before)


def test_alloc_pages_sees_pages_freed_but_not_yet_merged():
    allocator = _paged()
    allocator.alloc_pages(torch.tensor([6]))
    allocator.free_page_ids(torch.tensor([6]))  # staged, not merged yet
    assert allocator.alloc_pages(torch.tensor([6])) is not None


def test_token_allocator_alloc_pages():
    allocator = TokenToKVPoolAllocator(
        size=16,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=SimpleNamespace(),
        need_sort=False,
    )
    assert allocator.alloc_pages(torch.tensor([7, 3])).tolist() == [7, 3]
    assert allocator.alloc_pages(torch.tensor([7])) is None


@pytest.mark.parametrize("tree_core_backend", [None, "python"])
def test_import_prefix_serves_prefix_hits(tree_core_backend):
    allocator = _paged()
    cache = _cache(allocator, tree_core_backend)
    key = RadixKey(list(range(2 * PAGE)))

    assert cache.import_prefix(key, torch.tensor([5, 2])) == 2 * PAGE

    result = cache.match_prefix(MatchPrefixParams(key=key))
    assert result.device_prefix_len == 2 * PAGE
    path = cache.path_device_indices(result.last_device_node)
    assert path[: 2 * PAGE].tolist() == [20, 21, 22, 23, 8, 9, 10, 11]


def test_import_prefix_frees_pages_for_an_already_cached_prefix():
    allocator = _paged()
    cache = _cache(allocator)
    key = RadixKey(list(range(PAGE)))
    assert cache.import_prefix(key, torch.tensor([1])) == PAGE
    available = allocator.available_size()

    # Same content in another page: the tree keeps page 1, page 4 returns.
    assert cache.import_prefix(key, torch.tensor([4])) == PAGE
    assert allocator.available_size() == available
    assert 4 in allocator.get_all_free_pages().tolist()


def test_import_prefix_of_claimed_pages_does_not_claim_them_again():
    allocator = _paged()
    cache = _cache(allocator)
    allocator.alloc_pages(torch.tensor([2, 6]))
    available = allocator.available_size()
    key = RadixKey(list(range(2 * PAGE)))
    assert cache.import_prefix(key, torch.tensor([2, 6]), claimed=True) == 2 * PAGE
    assert allocator.available_size() == available
    result = cache.match_prefix(MatchPrefixParams(key=key))
    path = cache.path_device_indices(result.last_device_node)
    assert (path[::PAGE] // PAGE).tolist() == [2, 6]


def test_import_prefix_rejects_unavailable_pages_and_bad_keys():
    allocator = _paged()
    cache = _cache(allocator)
    allocator.alloc_pages(torch.tensor([3]))
    assert cache.import_prefix(RadixKey(list(range(PAGE))), torch.tensor([3])) == 0
    with pytest.raises(ValueError, match="pages of"):
        cache.import_prefix(RadixKey(list(range(PAGE + 1))), torch.tensor([2]))


if __name__ == "__main__":
    unittest.main()
