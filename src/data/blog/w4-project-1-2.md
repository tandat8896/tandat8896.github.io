---
title: "Project 1.2: Tạo và triển khai một chatbot cho một chủ đề cá nhân"
description: "Hướng dẫn chi tiết xây dựng và triển khai chatbot hỏi đáp tài liệu PDF cho một chủ đề cá nhân, sử dụng pipeline RAG, LLM nhỏ, chunking, embedding, QLoRA, LangChain, UI, ví dụ code thực tế."
pubDatetime: 2024-06-08T10:00:00Z
tags:
  - rag
  - llm
  - pdf
  - week4
draft: false
---

# Project 1.2: Tạo và triển khai một chatbot cho một chủ đề cá nhân

## Mục tiêu của project

Mục tiêu của project này là mình muốn xây dựng một hệ thống thử nghiệm để kiểm tra, đánh giá khả năng của chatbot RAG (Retrieval-Augmented Generation) trong việc trả lời câu hỏi dựa trên tài liệu PDF. Mình muốn kiểm thử các yếu tố như độ chính xác câu trả lời, khả năng truy xuất context phù hợp, tốc độ phản hồi, và tiềm năng mở rộng với các mô hình lớn hơn hoặc dữ liệu đa dạng hơn. Trong project này, mình chủ yếu triển khai dựa trên kiến thức kiểu "blackbox" – tức là mình sử dụng mô hình ngôn ngữ lớn (LLM) như một công cụ, không đi sâu vào cấu trúc bên trong hay quá trình huấn luyện của LLM, mà tập trung vào cách ứng dụng, tích hợp và kiểm thử LLM trong pipeline RAG. Đây là quá trình mình thử nghiệm, học hỏi nên rất mong nhận được sự thông cảm và góp ý từ mọi người. Cảm ơn mọi người nhiều.

---

## Lưu ý về phần cứng và lựa chọn mô hình trong project

Trong quá trình xây dựng project này, do hạn chế về phần cứng (GPU và CPU yếu), chỉ sử dụng các mô hình ngôn ngữ nhỏ, có sẵn trong codebase. Không thực hiện fine-tune hoặc triển khai các LLM lớn như Llama-2, Qwen-7B, v.v. Điều này giúp đảm bảo hệ thống có thể chạy được trên máy tính cá nhân cấu hình thấp, phù hợp cho mục đích học tập, thử nghiệm hoặc demo nhanh.

**Lưu ý:** Nếu muốn thử nghiệm với các mô hình lớn hơn, fine-tune hoặc sử dụng QLoRA, có thể triển khai project trên Google Colab để tận dụng GPU miễn phí hoặc trả phí. Colab hỗ trợ cài đặt các thư viện cần thiết, tải mô hình lớn và chạy inference/fine-tune hiệu quả hơn so với máy cá nhân cấu hình thấp.

- **Lý do lựa chọn:**
  - Phần cứng phổ thông không đủ RAM/GPU để chạy hoặc fine-tune các LLM lớn.
  - Ưu tiên tốc độ phản hồi và khả năng triển khai thực tế trên máy cá nhân.
  - Dễ dàng cài đặt, không yêu cầu môi trường phức tạp.

- **Ảnh hưởng:**
  - Chất lượng trả lời có thể không tốt bằng các LLM lớn hoặc đã fine-tune chuyên sâu.
  - Một số tính năng reasoning nâng cao, multi-step agent, hoặc trả lời phức tạp có thể bị hạn chế.

- **Giá trị thực tiễn:**
  - Dù dùng model nhỏ, hệ thống vẫn minh bạch, kiểm chứng được nguồn context.
  - Phù hợp cho các bài toán QA tài liệu vừa và nhỏ, hoặc làm nền tảng để mở rộng khi có phần cứng mạnh hơn.

---

## Giới thiệu về LangChain

LangChain là một framework mã nguồn mở giúp xây dựng các ứng dụng sử dụng mô hình ngôn ngữ lớn (LLM) một cách linh hoạt và hiệu quả. LangChain cung cấp các thành phần sẵn có để kết nối LLM với các nguồn dữ liệu (retriever), xây dựng pipeline hỏi đáp (QA), reasoning agent, tích hợp tool, và quản lý workflow phức tạp.

### Vai trò của LangChain trong hệ thống RAG
- Cho phép kết nối LLM với kho tri thức (vectorstore, database, API, file...)
- Hỗ trợ xây dựng các pipeline truy xuất - sinh (retrieval-augmented generation) nhanh chóng
- Dễ dàng tích hợp các agent reasoning, tool, memory, callback để mở rộng tính năng
- Quản lý workflow, debug, log reasoning từng bước

LangChain được sử dụng rộng rãi trong các project RAG hiện đại nhờ khả năng mở rộng, cộng đồng lớn, và hỗ trợ nhiều backend LLM khác nhau (OpenAI, HuggingFace, v.v.).

---

## QLoRA: Lý thuyết, vai trò, QUANTIZATION và thực tế tích hợp

### Lý thuyết QLoRA
**QLoRA (Quantized Low-Rank Adapter)** là một kỹ thuật fine-tune mô hình ngôn ngữ lớn (LLM) với chi phí bộ nhớ thấp nhờ lượng tử hóa (quantization) và chèn các adapter nhỏ (LoRA) vào các layer của LLM. QLoRA giúp:
- Fine-tune LLM trên dữ liệu riêng mà không cần GPU lớn.
- Giữ nguyên trọng số gốc của LLM, chỉ cập nhật các tham số nhỏ của adapter.
- Dễ dàng chuyển đổi giữa nhiều adapter cho các domain khác nhau.

### Nhấn mạnh: Quantization là gì và vì sao quan trọng?

**Quantization (Lượng tử hóa)** là quá trình chuyển các trọng số của mô hình từ dạng số thực 16-bit/32-bit (float16/float32) sang dạng số nguyên có độ dài bit thấp hơn (thường là 8-bit hoặc 4-bit). Đây là bước cốt lõi giúp QLoRA tiết kiệm bộ nhớ và tăng tốc độ xử lý.

- **Tại sao cần quantization?**
  - Mô hình LLM gốc rất lớn (hàng chục GB), khó fine-tune trên GPU phổ thông.
  - Quantization giảm kích thước mô hình (ví dụ: 4-bit giảm ~4-8 lần so với float32), cho phép fine-tune trên máy tính cá nhân hoặc cloud rẻ tiền.
  - Giảm chi phí lưu trữ và truyền tải mô hình.

- **Ảnh hưởng đến chất lượng:**
  - Nếu quantization quá thấp (ví dụ 2-bit), mô hình có thể mất nhiều thông tin, giảm chất lượng.
  - 4-bit quantization (như trong QLoRA) thường giữ được chất lượng gần như nguyên bản, nhưng tiết kiệm rất nhiều tài nguyên.

- **Các mức độ phổ biến:**
  - 8-bit: Dễ triển khai, chất lượng gần như không đổi.
  - 4-bit: Cân bằng tốt giữa tiết kiệm bộ nhớ và chất lượng, là lựa chọn mặc định của QLoRA.

- **Lưu ý thực tế:**
  - Một số GPU/CPU cũ không hỗ trợ tốt 4-bit quantization.
  - Khi inference, có thể dùng quantized model để tiết kiệm RAM, nhưng nếu cần chất lượng tối đa thì nên dùng bản full-precision.

### Vai trò trong project RAG
- Khi sử dụng LLM đã fine-tune bằng QLoRA trên tài liệu PDF hoặc domain riêng, chất lượng trả lời sẽ sát thực tế hơn, giảm "bịa" thông tin.
- QLoRA giúp cá nhân hóa LLM cho từng loại tài liệu, doanh nghiệp mà không tốn nhiều tài nguyên.
- Nhờ quantization, có thể triển khai LLM lớn cho RAG trên máy tính phổ thông hoặc cloud giá rẻ.

### Tích hợp QLoRA vào pipeline
- Nếu sử dụng mô hình LLM đã fine-tune bằng QLoRA, chỉ cần load adapter vào LLM trước khi gọi `generate` trong `qa_pipeline.py`.
- Ví dụ (giả lập):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('base-llm', load_in_4bit=True)  # Nhấn mạnh load_in_4bit
qlora_adapter = PeftModel.from_pretrained(base_model, 'qlora-adapter-path')
tokenizer = AutoTokenizer.from_pretrained('base-llm')

# Khi gọi LLM.generate, adapter QLoRA sẽ tự động được sử dụng
```

### Lưu ý thực tế khi dùng QLoRA và quantization
- Cần chọn base model phù hợp với task (QA, summarization, v.v.).
- Dữ liệu fine-tune phải chất lượng, sát với use-case thực tế.
- Nếu adapter QLoRA không phù hợp, LLM vẫn có thể "bịa" hoặc trả lời kém.
- Nên kiểm thử kỹ sau khi fine-tune để đảm bảo tính trung thực.
- Khi gặp lỗi về RAM/GPU, kiểm tra lại tham số quantization và phần cứng hỗ trợ.

---

## Chunking trong RAG: Lý thuyết, thực tế và ví dụ

### Lý thuyết chunking trong RAG

Chunking là quá trình chia nhỏ tài liệu thành các đoạn (chunk) để tăng hiệu quả truy xuất, giảm độ dài context truyền vào LLM, và giữ được ngữ nghĩa/thông tin quan trọng.

### Các kiểu chunking phổ biến và ví dụ

1. **Chunking theo ký tự (character-based)**
   - Chia text thành các đoạn có số ký tự cố định.
   - Ví dụ: Với chunk_size=40, overlap=10
     - Input: "Attention is all you need. The Transformer model uses self-attention for sequence modeling."
     - Output:
       - Chunk 1: "Attention is all you need. The Transformer mo"
       - Chunk 2: "former model uses self-attention for sequenc"
       - Chunk 3: "ion for sequence modeling."
   - Ưu điểm: Đơn giản, dễ cài đặt. Nhược điểm: Dễ cắt rời ngữ nghĩa, câu bị chia đôi.

2. **Chunking theo câu (sentence-based)**
   - Chia text thành các đoạn gồm N câu.
   - Ví dụ: Mỗi chunk gồm 2 câu.
     - Input: "Attention is all you need. The Transformer model uses self-attention. This allows each word to attend to all other words."
     - Output:
       - Chunk 1: "Attention is all you need. The Transformer model uses self-attention."
       - Chunk 2: "This allows each word to attend to all other words."
   - Ưu điểm: Giữ ngữ nghĩa tốt hơn. Nhược điểm: Cần thư viện tách câu.

3. **Chunking theo đoạn văn (paragraph-based)**
   - Mỗi chunk là một đoạn văn bản (ngăn cách bởi xuống dòng).
   - Ví dụ:
     - Input: "Attention is all you need.\nThe Transformer model uses self-attention."
     - Output:
       - Chunk 1: "Attention is all you need."
       - Chunk 2: "The Transformer model uses self-attention."
   - Ưu điểm: Phù hợp tài liệu có cấu trúc rõ ràng. Nhược điểm: Không tối ưu nếu đoạn quá dài/ngắn.

4. **Semantic chunking**
   - Chia theo ý nghĩa, chủ đề (dùng model phân đoạn semantic).
   - Ví dụ: Nếu đoạn nói về "self-attention" sẽ thành 1 chunk riêng.
   - Ưu điểm: Giữ ngữ nghĩa tốt nhất. Nhược điểm: Phức tạp, tốn tài nguyên.

### Ảnh hưởng của chunk_size và overlap
- chunk_size nhỏ: Tăng số chunk, truy xuất chính xác hơn nhưng context có thể thiếu ngữ cảnh rộng.
- chunk_size lớn: Giữ được nhiều ngữ cảnh, nhưng dễ bị loãng khi truy xuất.
- overlap lớn: Giảm nguy cơ mất thông tin ở ranh giới, nhưng tăng số chunk, tốn bộ nhớ.



### Gợi ý cải thiện
- Có thể dùng chunking theo câu hoặc đoạn văn để giữ ngữ nghĩa tốt hơn nếu tài liệu phù hợp.
- Tùy chỉnh chunk_size, overlap phù hợp từng loại tài liệu.
- Áp dụng semantic chunking nếu cần chất lượng context cao.

---

## Embedding, đo tương đồng và các ý tưởng mở rộng trong chunking

### Embedding và đo tương đồng (Cosine Similarity)
- Mỗi đoạn (chunk) sau khi chia sẽ được truyền qua embedding model để lấy vector đặc trưng.
- Khi có câu hỏi, hệ thống cũng lấy embedding cho câu hỏi.
- Độ tương đồng giữa câu hỏi và từng chunk được tính bằng **cosine similarity** (đã có hàm `cosine_sim` trong code).
- Các chunk có similarity cao nhất sẽ được chọn làm context trả lời.

### Ý tưởng cắt ngưỡng theo percentile (breakpoint threshold)
- Có thể loại bỏ các chunk có similarity thấp bằng cách lấy ngưỡng theo phân vị (percentile), ví dụ: chỉ giữ các chunk có similarity nằm trong top 95% (cắt 5% thấp nhất).
- Cách làm: Tính phân phối similarity, lấy điểm tại 95% làm threshold, loại bỏ các chunk thấp hơn ngưỡng này.
- **Hiện tại:** Chưa có sẵn trong code, có thể bổ sung nếu muốn tăng chất lượng context.

### Min chunk size, semantic chunking
- Chunk size mặc định là 500 ký tự, có thể tăng lên 600-700 nếu dùng semantic chunking để giữ ngữ nghĩa tốt hơn.
- Đã hỗ trợ chia chunk theo semantic, sentence, paragraph, sliding window...

### add_start_index (đánh chỉ mục chunk)
- Khi chia chunk, có thể thêm chỉ mục (start_index) vào metadata để biết vị trí từng đoạn trong tài liệu gốc.
- Điều này giúp truy vết, highlight, hoặc sắp xếp lại context khi cần.
- **Hiện tại:** Chưa có sẵn, có thể bổ sung vào quá trình chunking nếu cần.

---

## 2. Chunking và Similarity: Cắt đoạn và truy xuất context

### Lý thuyết
Trong hệ thống RAG, trước khi truy xuất context, tài liệu cần được chia nhỏ thành các đoạn (chunk) để tăng hiệu quả tìm kiếm. Sau đó, khi có câu hỏi, hệ thống sẽ đo similarity (độ tương đồng) giữa embedding của câu hỏi và embedding của từng đoạn để chọn ra context phù hợp nhất.

### Triển khai thực tế trong project

- **Chunking:**
  Văn bản PDF được cắt thành các đoạn có độ dài cố định bằng hàm `chunk_text`. Không có xử lý đặc biệt để chia theo câu, đoạn văn, hay semantic.

  ```python
  def chunk_text(text, chunk_size=500, overlap=50):
      chunks = []
      for i in range(0, len(text), chunk_size - overlap):
          chunks.append(text[i:i+chunk_size])
      return chunks
  ```
  > Việc cắt này chỉ đơn giản là lấy liên tiếp các ký tự, không đảm bảo ngữ nghĩa hoặc ranh giới câu.

  **Ví dụ thực tế:**
  - **Input:** Một đoạn text PDF: "Attention is all you need. The Transformer model uses self-attention."
  - **chunk_size=20, overlap=5**
  - **Output:**
    - Chunk 1: "Attention is all you "
    - Chunk 2: "is all you need. The T"
    - Chunk 3: "need. The Transformer "
    - Chunk 4: "The Transformer model "
    - Chunk 5: "model uses self-atten"
    - Chunk 6: "uses self-attention."

- **Similarity:**
  Khi người dùng đặt câu hỏi, hệ thống sẽ:
  1. Chuyển câu hỏi thành embedding vector.
  2. Đo similarity (thường là L2 distance hoặc cosine similarity) giữa embedding của câu hỏi và embedding của từng đoạn text đã cắt.
  3. Lấy ra các đoạn có similarity cao nhất làm context cho LLM.

  ```python
  def retrieve(self, query, top_k=3):
      query_emb = self.model.encode([query])
      D, I = self.index.search(np.array(query_emb), top_k)
      return [self.texts[i] for i in I[0]]
  ```
  > Việc đo similarity này hoàn toàn dựa trên vector hóa bằng model embedding, không có bước lọc ngữ nghĩa hoặc kiểm tra lại context.

  **Ví dụ thực tế:**
  - **Câu hỏi:** "Transformer dùng attention như thế nào?"
  - **Các chunk:** (như ví dụ trên)
  - **Embedding similarity:**
    - Chunk 4: 0.92 (cao nhất)
    - Chunk 5: 0.89
    - Chunk 3: 0.85
  - **Context trả về:** [Chunk 4, Chunk 5, Chunk 3]

- **Đánh giá thực tế:**
  - Đơn giản, dễ triển khai, phù hợp cho project nhỏ hoặc thử nghiệm.
  - Việc cắt đoạn cố định có thể làm mất ngữ nghĩa, hoặc context bị chia cắt không hợp lý.
  - Nếu câu hỏi liên quan đến thông tin nằm ở ranh giới hai đoạn, có thể không truy xuất đủ context.

---

## 3. file_loader.py- Đọc và chia nhỏ PDF

**Lý thuyết:**  
Cần chia nhỏ tài liệu thành các đoạn (chunk) hợp lý để truy xuất hiệu quả, tránh mất thông tin ở ranh giới.

**Triển khai thực tế:**
```python
from PyPDF2 import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
```
- **Tham số:**
  - `chunk_size`, `overlap` giúp kiểm soát độ chi tiết và liền mạch của context.

**Ví dụ thực tế:**
- **Input:** File PDF gồm 2 trang, mỗi trang chứa 1 đoạn text.
- **Output:**
  - Text sau khi đọc: "Attention is all you need.\nThe Transformer model uses self-attention."

**Điểm mạnh:**
- Đọc được toàn bộ nội dung PDF.
- Có tham số `chunk_size` và `overlap` để điều chỉnh.

**Điểm hạn chế:**
- Hàm `extract_text()` của PyPDF2 đôi khi trả về None hoặc thiếu text (với PDF scan hoặc nhiều cột).
- Chưa xử lý loại bỏ ký tự thừa, xuống dòng, hoặc các trường hợp text bị dính liền.
- Chưa kiểm tra trường hợp file PDF không có text.

**Gợi ý cải thiện:**
- Thêm bước làm sạch text (`clean_text`).
- Kiểm tra và bỏ qua các page không có text.
- Có thể dùng các thư viện OCR nếu PDF là scan.

---

## 4. knowledge_base.py- Tạo và truy vấn vectorstore

**Lý thuyết:**  
Mỗi chunk được chuyển thành vector embedding, lưu vào vectorstore (FAISS). Khi có câu hỏi, cũng chuyển thành vector và tìm các chunk gần nhất.

**Triển khai thực tế:**
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class KnowledgeBase:
    def __init__(self, embedding_model="BAAI/bge-small-en"):
        self.model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(768)
        self.texts = []

    def add_chunks(self, chunks):
        embeddings = self.model.encode(chunks)
        self.index.add(np.array(embeddings))
        self.texts.extend(chunks)

    def retrieve(self, query, top_k=3):
        query_emb = self.model.encode([query])
        D, I = self.index.search(np.array(query_emb), top_k)
        return [self.texts[i] for i in I[0]]
```
- **Tham số:**
  - `embedding_model` quyết định cách so khớp ngữ nghĩa.
  - `top_k` kiểm soát số lượng context trả về (nhiều quá có thể loãng, ít quá có thể thiếu).

**Ví dụ thực tế:**
- **Input:**
  - Các chunk: ["Attention is all you ", "is all you need. The T", ...]
  - Câu hỏi: "Transformer dùng attention như thế nào?"
- **Output:**
  - Embedding của từng chunk và câu hỏi được lưu vào FAISS.
  - Truy vấn trả về các chunk có similarity cao nhất.

**Điểm mạnh:**
- Sử dụng FAISS cho truy vấn nhanh.
- Dễ thay đổi model embedding.

**Điểm hạn chế:**
- Cố định dimension 768, nếu đổi model khác có thể lỗi.
- Không lưu metadata (ví dụ: số trang, vị trí chunk), nên không biết context nằm ở đâu trong tài liệu.
- Chưa kiểm tra trường hợp số chunk ít hơn `top_k`.
- Chưa chuẩn hóa embedding (nên dùng cosine similarity thay vì L2 cho nhiều model).

**Gợi ý cải thiện:**
- Lưu thêm metadata cho mỗi chunk.
- Kiểm tra dimension của embedding.
- Cho phép chọn loại similarity (cosine/L2).
- Bắt lỗi khi số chunk < top_k.

---

## 5. qa_pipeline.py – Pipeline hỏi đáp

**Lý thuyết:** LLM chỉ nên trả lời dựa trên context đã truy xuất, không tự ý "bịa" thông tin ngoài context.

**Triển khai thực tế:**
```python
class QAPipeline:
    def __init__(self, knowledge_base, llm):
        self.kb = knowledge_base
        self.llm = llm

    def answer(self, question, top_k=3, debug=False):
        context = self.kb.retrieve(question, top_k=top_k)
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        answer = self.llm.generate(prompt)
        if debug:
            return answer, context
        return answer
```
- **Tham số:**
  - `top_k` (số context), `debug` (trả về context để kiểm chứng).

**Ví dụ thực tế:**
- **Input:**
  - Câu hỏi: "Transformer dùng attention như thế nào?"
  - Context: ["The Transformer model uses self-attention.", ...]
- **Prompt truyền vào LLM:**
  ```
  Context: ['The Transformer model uses self-attention.', ...]
  Question: Transformer dùng attention như thế nào?
  Answer:
  ```
- **Output:**
  - "Transformer sử dụng cơ chế self-attention để mỗi từ trong câu có thể tập trung vào các từ khác, giúp mô hình hiểu ngữ cảnh tốt hơn."

**Điểm mạnh:**
- Luôn truyền context vào prompt.
- Có chế độ debug để kiểm tra context.

**Điểm hạn chế:**
- Prompt chưa kiểm soát chặt chẽ việc LLM chỉ trả lời dựa trên context (LLM vẫn có thể "bịa" nếu prompt không đủ rõ).
- Chưa có cơ chế kiểm tra LLM có dùng đúng context không.
- Nếu context quá dài, prompt có thể vượt quá giới hạn token của LLM.

**Gợi ý cải thiện:**
- Thêm hướng dẫn rõ ràng trong prompt: "Chỉ trả lời dựa trên context, nếu không có thì trả lời 'Không tìm thấy thông tin'."
- Cắt bớt context nếu quá dài.
- Có thể log lại prompt và answer để kiểm tra chất lượng.

---

## 6. ui.py- Giao diện Streamlit

**Lý thuyết:** Người dùng cần kiểm tra được nguồn gốc thông tin, điều chỉnh tham số để kiểm soát chất lượng câu trả lời.

**Triển khai thực tế:**
```python
import streamlit as st

def main_ui(pipeline):
    st.title("PDF RAG QA")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    question = st.text_input("Ask a question:")
    top_k = st.sidebar.slider("Top K", 1, 10, 3)
    debug = st.sidebar.checkbox("Show debug info")

    if uploaded_file and question:
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        kb = KnowledgeBase()
        kb.add_chunks(chunks)
        qa = QAPipeline(kb, llm=YourLLM())
        answer, context = qa.answer(question, top_k=top_k, debug=debug)
        st.write("**Answer:**", answer)
        if debug:
            with st.expander("Context"):
                st.write(context)
```
- Có thể điều chỉnh `top_k` để kiểm soát lượng context.

**Ví dụ thực tế:**
- **Input:**
  - File PDF: "Attention is all you need. The Transformer model uses self-attention."
  - Câu hỏi: "Transformer dùng attention như thế nào?"
- **Output trên UI:**
  - **Answer:** "Transformer sử dụng cơ chế self-attention để mỗi từ trong câu có thể tập trung vào các từ khác, giúp mô hình hiểu ngữ cảnh tốt hơn."
  - **Context (debug):** ["The Transformer model uses self-attention.", ...]

**Điểm mạnh:**
- Cho phép điều chỉnh `top_k`, bật/tắt debug.
- Hiển thị context để kiểm chứng.

**Điểm hạn chế:**
- Mỗi lần hỏi lại phải nạp lại toàn bộ file, không lưu knowledge base giữa các lần hỏi (tốn thời gian).
- Không hiển thị vị trí context trong tài liệu.
- Không có thông báo lỗi nếu file PDF không hợp lệ.

**Gợi ý cải thiện:**
- Lưu knowledge base vào session hoặc cache.
- Hiển thị thêm metadata (số trang, vị trí chunk).
- Bắt lỗi khi file PDF không đọc được.

---

## 7. utils.py - Các hàm tiện ích

**Lý thuyết:** Tiền xử lý giúp đảm bảo dữ liệu đầu vào sạch, không bị nhiễu, tăng độ chính xác khi truy xuất.

**Triển khai thực tế:**
```python
def clean_text(text):
    return text.replace('\n', ' ').strip()
```

**Ví dụ thực tế:**
- **Input:** "Attention is all you need.\nThe Transformer model uses self-attention."
- **Output:** "Attention is all you need. The Transformer model uses self-attention."

**Điểm mạnh:**  
- Đơn giản, dễ dùng.

**Điểm hạn chế:**  
- Chưa xử lý các ký tự đặc biệt, nhiều khoảng trắng, hoặc các lỗi OCR.

**Gợi ý cải thiện:**  
- Thêm loại bỏ ký tự đặc biệt, chuẩn hóa unicode, tách câu.

---

## 8. workflow.py -  Điều phối tổng thể

**Lý thuyết:** Đảm bảo mọi bước đều dựa trên dữ liệu thực tế, không có "shortcut" nào bỏ qua truy xuất context.

**Triển khai thực tế:**
```python
def run_workflow():
    main_ui(pipeline=QAPipeline)
```

**Ví dụ thực tế:**
- **Input:** Người dùng upload file PDF, nhập câu hỏi, nhấn gửi.
- **Output:** UI hiển thị answer và context như các ví dụ trên.

**Điểm mạnh:**  
- Đơn giản, dễ mở rộng.

**Điểm hạn chế:**  
- Chưa có kiểm soát lỗi tổng thể, chưa hỗ trợ nhiều workflow khác nhau.

---

## Vai trò của StrOutputParser trong pipeline RAG

Trong pipeline RAG, sau khi context và câu hỏi được kết hợp thành prompt và gửi qua LLM, kết quả trả về từ LLM thường ở dạng thô (raw output). Để đảm bảo kết quả này luôn ở dạng string chuẩn, dễ xử lý và hiển thị cho người dùng, ta sử dụng **StrOutputParser**.

**StrOutputParser** chính là phần kết nối quan trọng giữa LLM và ứng dụng:
- Nhận output thô từ LLM, chuyển đổi (parse) thành dạng string chuẩn.
- Đảm bảo output luôn đúng định dạng, giúp ứng dụng (ví dụ: Streamlit UI) dễ dàng hiển thị kết quả.
- Nếu không có bước này, output từ LLM có thể ở dạng phức tạp (object, dict, v.v.), gây lỗi hoặc khó hiển thị.

**Tóm lại:** StrOutputParser là cầu nối giữa LLM và phần hiển thị kết quả, giúp pipeline mạch lạc, dễ debug và mở rộng. Nếu muốn custom logic phân tích output, chỉ cần thay thế hoặc mở rộng chỗ này trong code.


