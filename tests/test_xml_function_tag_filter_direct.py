#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Direct tests for XMLFunctionTagFilter without pipeline machinery."""

import pytest
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter


@pytest.mark.asyncio
async def test_filter_removes_function_calls():
    """Test that the filter removes function call syntax."""
    
    xml_filter = XMLFunctionTagFilter()
    
    original_text = (
        "Hello! I can help you schedule that interview. "
        "<function=schedule_interview>{\"date\": \"tomorrow\", \"time\": \"3 PM\"}</function> "
        "Your interview has been successfully scheduled."
    )
    
    filtered_text = await xml_filter.filter(original_text)
    
    # Verify function call syntax was removed
    assert "<function" not in filtered_text, f"Function syntax found: {filtered_text}"
    assert "</function>" not in filtered_text, f"Function syntax found: {filtered_text}"
    assert "function=" not in filtered_text, f"Function syntax found: {filtered_text}"
    assert "{\"date\":" not in filtered_text, f"JSON found in filtered text: {filtered_text}"
    
    # Verify meaningful content was preserved  
    assert "Hello!" in filtered_text, f"Content missing: {filtered_text}"
    assert "interview" in filtered_text, f"Content missing: {filtered_text}"
    assert "scheduled" in filtered_text, f"Content missing: {filtered_text}"
    
    print(f"Original: {original_text}")
    print(f"Filtered: {filtered_text}")
    print(f"Filter test successful")


@pytest.mark.asyncio
async def test_filter_with_multiple_function_calls():
    """Test filtering multiple function calls in one text."""
    
    xml_filter = XMLFunctionTagFilter()
    
    original_text = (
        "Starting the call <function=always_move_to_main_agenda></function> "
        "with agenda items. <function=end_call></function> Call ended."
    )
    
    filtered_text = await xml_filter.filter(original_text)
    
    # Verify all function calls were removed
    assert "<function" not in filtered_text
    assert "</function>" not in filtered_text
    assert "function=" not in filtered_text
    
    # Verify meaningful content preserved
    assert "Starting the call" in filtered_text
    assert "with agenda items" in filtered_text
    assert "Call ended" in filtered_text
    
    print(f"Multiple function calls test passed")
    print(f"Filtered: {filtered_text}")


@pytest.mark.asyncio
async def test_filter_preserves_normal_text():
    """Test that normal text passes through unchanged."""
    
    xml_filter = XMLFunctionTagFilter()
    
    original_text = "Hello! Your interview has been successfully scheduled for tomorrow."
    
    filtered_text = await xml_filter.filter(original_text)
    
    # Should be unchanged
    assert filtered_text == original_text
    
    print(f"Normal text preservation test passed")


@pytest.mark.asyncio 
async def test_filter_handles_empty_text():
    """Test that empty text is handled correctly."""
    
    xml_filter = XMLFunctionTagFilter()
    
    test_cases = [
        "",  # Empty string
        "   ",  # Only whitespace
        "<function=test></function>",  # Only function call
    ]
    
    for original_text in test_cases:
        filtered_text = await xml_filter.filter(original_text)
        
        # Should not contain function syntax
        assert "<function" not in filtered_text
        assert "</function>" not in filtered_text
        
    print(f"Empty text handling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])