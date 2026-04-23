from docx import Document

def docx_to_txt(input_file, output_file):
    try:
        # Load the Word document
        doc = Document(input_file)

        # Extract all text
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)

        # Join text with line breaks
        text_content = "\n".join(full_text)

        # Write to txt file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text_content)

        print(f"✅ Successfully converted '{input_file}' to '{output_file}'")

    except Exception as e:
        print(f"❌ Error: {e}")


# Example usage
input_path = r"C:\Users\PC\Downloads\OrderCircle Knowledge Base.docx"
output_path = r"C:\Users\PC\Downloads\OrderCircle Knowledge Base output.txt"

docx_to_txt(input_path, output_path)