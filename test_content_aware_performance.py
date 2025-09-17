"""Quick test for ContentAwareChunker performance."""

from raglib.techniques.content_aware_chunker import ContentAwareChunker
from raglib.schemas import Document
import time

print('Testing ContentAwareChunker performance...')

# Test with the same text that was causing issues
test_text = (
    'This is the first paragraph. It has some content.\n\n'
    'This is the second paragraph. It also has content that should '
    'be kept together when possible.\n\n'
    'This is a third paragraph.'
)

chunker = ContentAwareChunker(max_chunk_size=100, min_chunk_size=20)
doc = Document(id='test_doc', text=test_text)

start_time = time.time()
result = chunker.apply(doc)
end_time = time.time()

print(f'Time taken: {end_time - start_time:.3f} seconds')
print(f'Success: {result.success}')
print(f'Number of chunks: {len(result.payload["chunks"])}')

if result.success:
    for i, chunk in enumerate(result.payload['chunks']):
        print(f'Chunk {i}: {len(chunk.text)} chars - "{chunk.text[:50]}..."')