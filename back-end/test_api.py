import requests
import json

def test_base_endpoint():
    response = requests.get("http://localhost:8000/")
    print("Base endpoint test:", response.text)
    assert response.status_code == 200

def test_rag_endpoint():
    # Test với câu hỏi về BHXH
    test_questions = [
        "Cho tôi biết về chế độ thai sản",
        "Điều kiện hưởng BHXH một lần là gì?",
        "Mức đóng BHXH hiện nay là bao nhiêu?"
    ]
    
    for question in test_questions:
        print(f"\nTesting question: {question}")
        response = requests.get(f"http://localhost:8000/rag", params={"q": question})
        
        if response.status_code == 200:
            result = response.json()
            print("Status: Success")
            print("Answer:", result["result"])
            print("Number of sources:", len(result["source_documents"]))
        else:
            print("Status: Failed")
            print("Error:", response.text)

if __name__ == "__main__":
    print("Starting API tests...")
    try:
        test_base_endpoint()
        print("\nBase endpoint test passed!")
        
        print("\nTesting RAG endpoint...")
        test_rag_endpoint()
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}") 