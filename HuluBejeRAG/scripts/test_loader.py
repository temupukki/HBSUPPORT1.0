from app.ingestion.loader import DocumentLoader


loader = DocumentLoader("knowledge")

documents = loader.load()

print(f"Documents loaded: {len(documents)}")

for document in documents:

    print()
    print("=" * 60)
    print(f"Document: {document.document_id}")
    print(f"Source: {document.source}")
    print(f"Pages: {document.page_count}")
    print("=" * 60)

    for page in document.pages[:2]:

        print()
        print(f"Page {page.page_number}")
        print("-" * 40)
        print(page.text[:500])