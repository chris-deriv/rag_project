from src.documents import DocumentProcessor

def main():
    print("Processing test.pdf...")
    # Create processor with improved settings
    processor = DocumentProcessor(
        chunk_size=150,  # Even smaller chunks to respect section boundaries
        chunk_overlap=50,  # Larger overlap to avoid splitting sentences
        length_function="char"  # Character count for predictable chunks
    )
    
    # Override the text splitter's separators to better respect document structure
    processor.text_splitter.separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ".",     # Sentence breaks
        "!",     # Exclamation marks
        "?",     # Question marks
        ";",     # Semicolons
        ":",     # Colons
        " ",     # Words
        ""       # Characters
    ]
    
    result = [
        {
            "id": chunk.id,
            "text": chunk.text.strip(),  # Clean up any leading/trailing whitespace
            **chunk.metadata
        }
        for chunk in processor.process_document('uploads/test.pdf')
    ]
    
    print(f"\nFound {len(result)} chunks:\n")
    for i, chunk in enumerate(result, 1):
        print(f"Chunk {i}:")
        print(f"Text: {chunk['text']}")
        print(f"Metadata:")
        for key, value in sorted(chunk.items()):  # Sort metadata keys for readability
            if key != 'text':
                print(f"  {key}: {value}")
        print()

if __name__ == '__main__':
    main()
