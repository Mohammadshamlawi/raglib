#!/usr/bin/env python
"""Test script for new chunking techniques."""

from raglib.techniques import (
    ContentAwareChunker, 
    DocumentSpecificChunker, 
    RecursiveChunker, 
    PropositionalChunker,
    ParentDocumentChunker
)
from raglib.schemas import Document

print('Testing all new chunking techniques...')

# Test document
doc = Document(
    id='test', 
    text='This is a test document. It has multiple sentences. Each sentence should be handled properly by the chunking techniques.'
)

# Test each technique
techniques = [
    ('ContentAwareChunker', ContentAwareChunker()),
    ('DocumentSpecificChunker', DocumentSpecificChunker()),
    ('RecursiveChunker', RecursiveChunker()),
    ('PropositionalChunker', PropositionalChunker()),
    ('ParentDocumentChunker', ParentDocumentChunker())
]

for name, chunker in techniques:
    try:
        result = chunker.apply(doc)
        if result.success:
            chunks = result.payload.get('chunks', [])
            print(f'✓ {name}: {len(chunks)} chunks created')
        else:
            error_msg = result.payload.get('error', 'Unknown error')
            print(f'✗ {name}: Failed - {error_msg}')
    except Exception as e:
        print(f'✗ {name}: Exception - {str(e)}')

print('All tests completed!')