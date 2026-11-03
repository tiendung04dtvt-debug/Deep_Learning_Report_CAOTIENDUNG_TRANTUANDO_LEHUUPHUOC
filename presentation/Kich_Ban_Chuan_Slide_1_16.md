# 🎙️ KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỀ TÀI (SLIDE 1 ➔ SLIDE 16)
**Đề tài**: Xây dựng & tối ưu hóa mô hình học sâu (YOLOv8 + Custom Deep CNN) trong hệ thống giám sát giao thông và nhận diện biển số xe thời gian thực  
**GVHD**: TS. Trần Thị Minh Hạnh  
**Nhóm sinh viên**: Cao Tiến Dũng, Trần Tuấn Đô (22DT2) · Lê Hữu Phước (22DT4)  
**Phạm vi Người 1**: Từ Slide 1 đến Slide 16 (Bản slide mới: `Dual_AI_Bien_so_xe_V2 (2).pptx`)  
**Tổng thời lượng chuẩn xác**: **Đúng 6 phút 00 giây (360 giây)**  

---

### BẢNG TRA NHANH THỜI GIAN & ĐIỂM CHỐT TỪNG SLIDE (TỔNG: 360 GIÂY = 6:00)

| Slide | Tiêu đề | Mốc thời gian | Thời lượng | Điểm then chốt cần nhớ |
| :---: | :--- | :---: | :---: | :--- |
| **01** | Báo cáo đề tài | 0:00 – 0:20 | 20s | Chào trang trọng, nêu đúng tên GVHD & 3 thành viên (Dũng, Đô, Phước). |
| **02** | Đặt vấn đề & mục tiêu | 0:20 – 0:45 | 25s | Biển số < 5% khung hình, chói sáng/bóng râm ➔ Mục tiêu hệ thống. |
| **03** | Kiến trúc Dual-AI (Tổng quan) | 0:45 – 1:05 | 20s | Phân tầng: YOLOv8 lọc không gian + Bộ giải mã ký tự. |
| **04** | Kiến trúc Dual-AI (Pipeline) | 1:05 – 1:35 | 30s | Quy hoạch 4 giai đoạn: G1 (YOLOv8/Tracking) ➔ G2 (Tiền xử lý) ➔ G3 (PlateCharCNN) ➔ G4 (Hybrid). |
| **05** | Hành trình khung hình | 1:35 – 1:55 | 20s | 5 bước tuần tự, lấy ví dụ xe #12 đi từ ảnh thô ra biển 51H-049.64. |
| *Tiểu kết* | *Mở đầu & Kiến trúc (Slide 1–5)* | *0:00 – 1:55* | *115s* | *Khởi đầu rõ ràng, tự tin, đúng tiến độ 1 phút 55 giây.* |
| **06** | YOLOv8 phát hiện xe | 1:55 – 2:10 | 15s | 3 đặc trưng: Anchor-free, khối C2f, tốc độ thời gian thực. |
| **07** | Cơ chế phát hiện xe | 2:10 – 2:30 | 20s | 4 giai đoạn: ảnh 640×640, tích chập viền xe, dự đoán ô lưới, `yolov8n.pt`. |
| **08** | Minh họa nhận diện xe | 2:30 – 2:45 | 15s | Phát hiện đồng thời nhiều xe trên làn đường với độ tin cậy ổn định. |
| **09** | Kalman Filter & IoU (BoT-SORT) | 2:45 – 3:10 | 25s | Kalman dự đoán chuyển động, `fuse_score: True`, `match_thresh: 0.8`. |
| **10** | Minh họa bám đuôi | 3:10 – 3:25 | 15s | Quỹ đạo mượt mà, khắc phục tình trạng che khuất tạm thời. |
| *Tiểu kết* | *Bước 1: Bám đuôi xe (Slide 6–10)* | *1:55 – 3:25* | *90s* | *Hoàn thành trọn vẹn Bước 1 trong đúng 1 phút 30 giây.* |
| **11** | Cắt biển & ghép xe | 3:25 – 3:50 | 25s | YOLOv8 biển số ➔ Hungarian Matching (`linear_sum_assignment`) tối ưu 1-1. |
| **12** | Quy trình xây dựng YOLOv8 | 3:50 – 4:05 | 15s | Sơ đồ 4 bước: Dữ liệu ➔ Kiến trúc ➔ Hàm mất mát ➔ Lịch trình học. |
| **13** | Thu thập dữ liệu | 4:05 – 4:30 | 25s | 8,000 ảnh phương tiện + 2,500 ảnh VN, tỷ lệ 80% Train / 20% Val. |
| **14** | Kiến trúc mạng YOLOv8 | 4:30 – 5:00 | 30s | C2f/SPPF Backbone ➔ PAN-FPN Neck (P3 80×80) ➔ Decoupled Head. |
| **15** | Các hàm mất mát | 5:00 – 5:30 | 30s | Trọng số: Box Loss CIoU (7.5) + DFL viền mịn (1.5) + BCE phân loại (0.5). |
| **16** | Lịch trình học & LR | 5:30 – 6:00 | 30s | LR tăng đến Ep 9 rồi giảm dần ➔ `close_mosaic: 10` ➔ Peak mAP 75.15% ➔ Bàn giao. |
| *Tiểu kết* | *Bước 2: Xây dựng YOLOv8 (Slide 11–16)* | *3:25 – 6:00* | *155s* | *Hoàn thành trọn vẹn Bước 2 trong đúng 2 phút 35 giây.* |
| **TỔNG CỘNG** | **TOÀN BỘ SLIDE 1 ➔ 16** | **0:00 – 6:00** | **360s** | **Đúng chuẩn 6 phút 00 giây (115s + 90s + 155s = 360s)** |

---

## 📜 NỘI DUNG LỜI THOẠI CHI TIẾT TỪNG SLIDE

### SLIDE 1: BÁO CÁO ĐỀ TÀI (Trang bìa)
- **Thời lượng**: 0:00 – 0:20 (20 giây).
- **Thao tác**: Đứng thẳng, mắt hướng về Hội đồng, mỉm cười tự tin, cúi đầu chào trang trọng.
- **Lời thoại**:
  > *"Kính thưa quý Thầy Cô trong Hội đồng đánh giá và các bạn sinh viên.*  
  > *Em xin đại diện nhóm báo cáo đề tài: **'Xây dựng & tối ưu hóa mô hình học sâu (YOLOv8 + Custom Deep CNN) trong hệ thống giám sát giao thông và nhận diện biển số xe thời gian thực'**, dưới sự hướng dẫn khoa học của **TS. Trần Thị Minh Hạnh**.*  
  > *Nhóm nghiên cứu gồm 3 thành viên: **Cao Tiến Dũng, Trần Tuấn Đô** (Lớp 22DT2) và **Lê Hữu Phước** (Lớp 22DT4).*  
  > *Sau đây, em xin phép bắt đầu phần trình bày về bối cảnh, kiến trúc hệ thống và quy trình xây dựng mô hình định vị biển số."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & MỤC TIÊU
- **Thời lượng**: 0:20 – 0:45 (25 giây).
- **Thao tác**: Hướng tay sang cột Thách thức ➔ Chuyển sang cột Mục tiêu đề tài bên phải.
- **Lời thoại**:
  > *"Trong bài toán giám sát phương tiện tự động tại các trạm kiểm soát, hệ thống thị giác máy tính phải giải quyết 3 thách thức thực tế:*  
  > *• Thứ nhất, diện tích vùng biển số rất nhỏ, **chiếm dưới 5% diện tích khung hình**.*  
  > *• Thứ hai, các yếu tố nhiễu môi trường như **chói đèn pha ban đêm, bóng râm và bụi bẩn** làm suy giảm chất lượng ảnh.*  
  > *• Thứ ba, hệ thống đòi hỏi sự cân bằng giữa **độ chính xác định vị** và **tốc độ xử lý khung hình**.*  
  > *Từ đó, mục tiêu của đề tài là xây dựng một chuỗi xử lý hoàn chỉnh: ứng dụng YOLOv8 phát hiện phương tiện và biển số, kết hợp mô hình nhận dạng ký tự, tinh chỉnh siêu tham số và tích hợp hệ thống dịch vụ với FastAPI, SQLite và WebSocket Dashboard."*

---

### SLIDE 3: KIẾN TRÚC HỆ THỐNG DUAL-AI (Sơ đồ khối tổng quan)
- **Thời lượng**: 0:45 – 1:05 (20 giây).
- **Thao tác**: Chỉ sơ đồ khối từ Camera ➔ YOLOv8 ➔ Tiền xử lý ➔ Nhận diện ký tự ➔ Dashboard.
- **Lời thoại**:
  > *"Để giải quyết bài toán trên, kiến trúc hệ thống được thiết kế phân tầng thành hai khối chính:*  
  > *• Khối thứ nhất là **Mô hình định vị YOLOv8n**: Đóng vai trò chọn lọc không gian, khoanh vùng phương tiện và bóc tách khu vực biển số, loại bỏ phần lớn diện tích ngoại cảnh không cần thiết.*  
  > *• Khối thứ hai là **Cụm tiền xử lý và nhận diện ký tự**: Hình ảnh biển số sau khi trích xuất được nâng cao chất lượng và đưa vào mô hình phân loại ký tự, kết hợp với các quy tắc định dạng biển số Việt Nam.*  
  > *Sự phân tách này giúp tối ưu hóa tài nguyên tính toán cho từng công đoạn riêng biệt."*

---

### SLIDE 4: KIẾN TRÚC HỆ THỐNG DUAL-AI (Sơ đồ luồng kỹ thuật thiết kế)
- **Thời lượng**: 1:05 – 1:35 (30 giây).
- **Thao tác**: Dùng bút chỉ lia theo thứ tự 4 khối màu từ trái sang phải: Khối xanh lam ➔ Khối tím ➔ Khối tím đậm ➔ Khối xanh lá ➔ Hộp kết quả đen.
- **Lời thoại**:
  > *"Thưa Thầy Cô, trên Slide 4 là sơ đồ luồng kỹ thuật tổng thể được nhóm quy hoạch qua **4 giai đoạn liên hoàn** từ luồng video đầu vào đến cơ sở dữ liệu đầu ra:*  
  > *• **Giai đoạn 1 (Khối xanh lam):** Mô hình YOLOv8 kết hợp bộ lọc Kalman và thuật toán bám vết phương tiện, đồng thời định vị vùng biển số xe.*  
  > *• **Giai đoạn 2 (Khối tím):** Cầu nối tiền xử lý ảnh sử dụng biên đệm an toàn 14 pixel, cân bằng sáng thích ứng CLAHE và nhị phân hóa Otsu để bóc tách từng ký tự $32 \times 32$.*  
  > *• **Giai đoạn 3 (Khối tím đậm):** Mạng tích chập chuyên sâu PlateCharCNN với 3 khối Conv-BN-Pool để phân loại 36 lớp ký tự.*  
  > *• **Giai đoạn 4 (Khối xanh lá):** Cơ chế ra quyết định kết hợp bộ đối soát định dạng biển số Việt Nam, kênh EasyOCR dự phòng và thuật toán bỏ phiếu đa khung hình để cho ra kết quả cuối cùng trong khoảng 52 ms.*  
  > *Trong phần báo cáo của em, em xin phép đi sâu vào **Giai đoạn 1 và quy trình huấn luyện mô hình định vị biển số**, còn **Giai đoạn 2, 3 và 4** sẽ được bạn [Tên Bạn 2] trình bày chi tiết ở phần tiếp theo."*

---

### SLIDE 5: HÀNH TRÌNH CỦA MỘT KHUNG HÌNH (Minh họa quy trình 5 bước)
- **Thời lượng**: 1:35 – 1:55 (20 giây).
- **Thao tác**: Chỉ 5 bước tuần tự từ 1 đến 5, nhấn mạnh vào dòng ví dụ thực tế xe #12 ở đáy slide.
- **Lời thoại**:
  > *"Dữ liệu từ khung hình camera được xử lý qua 5 bước tuần tự, minh họa qua phương tiện mang mã theo dõi #12:*  
  > *• **Bước 1:** Phát hiện thân xe và theo dõi chuyển động ➔ Thu được hộp bao và Track ID #12.*  
  > *• **Bước 2:** Định vị biển số và ghép đúng vào thân xe #12 bằng thuật toán Hungarian.*  
  > *• **Bước 3:** Cắt ảnh biển số và thực hiện các bước tiền xử lý nâng cao chất lượng.*  
  > *• **Bước 4:** Nhận diện các ký tự trên biển số ➔ Thu được chuỗi ký tự thô.*  
  > *• **Bước 5:** Áp dụng các quy tắc kiểm tra định dạng ➔ Chuẩn hóa thành biển số hoàn chỉnh: **51H-049.64**.*  
  > *Sau đây, em xin trình bày chi tiết về **Bước 1 và Bước 2** của hệ thống."*

---

### SLIDE 6: YOLOV8 PHÁT HIỆN XE (Bước 1 - 3 Đặc trưng chính)
- **Thời lượng**: 1:55 – 2:10 (15 giây).
- **Thao tác**: Hướng tay vào 3 hình ảnh minh họa trên slide.
- **Lời thoại**:
  > *"Ở Bước 1, nhiệm vụ đầu tiên là phát hiện phương tiện lưu thông trên làn đường. Nhóm sử dụng mạng **YOLOv8** nhờ các ưu điểm kiến trúc:*  
  > *1. Cơ chế **Anchor-free**: Dự đoán trực tiếp khoảng cách từ tâm đối tượng thay vì phụ thuộc vào tập hộp neo định sẵn.*  
  > *2. Khối cấu trúc **C2f**: Tối ưu đường truyền gradient và giảm tải tính toán.*  
  > *3. Tốc độ thực thi cao, phù hợp với yêu cầu xử lý liên tục các khung hình video."*

---

### SLIDE 7: YOLOV8 PHÁT HIỆN XE (Cơ chế phát hiện 4 giai đoạn)
- **Thời lượng**: 2:10 – 2:30 (20 giây).
- **Thao tác**: Chỉ lần lượt 4 khối nội dung từ 1 đến 4.
- **Lời thoại**:
  > *"Quá trình phát hiện phương tiện diễn ra qua 4 giai đoạn cơ bản:*  
  > *• Khung hình đầu vào được điều chỉnh kích thước về $640 \times 640$ và chia thành lưới các ô vuông.*  
  > *• Mạng tích chập sâu trích xuất các đặc trưng hình học như viền thân xe, nắp ca-pô và bánh xe.*  
  > *• Tại mỗi ô lưới, mạng dự đoán đồng thời xác suất có vật thể, tọa độ hộp bao và phân loại dòng xe (ô tô, xe tải, xe buýt, xe máy).*  
  > *• Về mô hình, nhóm sử dụng trọng số nền tảng **yolov8n.pt** đã được huấn luyện sẵn trên tập dữ liệu COCO, đáp ứng tốt việc nhận dạng các phương tiện phổ biến."*

---

### SLIDE 8: YOLOV8 PHÁT HIỆN XE (Minh họa nhận diện thực tế)
- **Thời lượng**: 2:30 – 2:45 (15 giây).
- **Thao tác**: Chỉ vào hình ảnh kết quả phát hiện xe trên mặt đường.
- **Lời thoại**:
  > *"Hình ảnh trên slide minh họa kết quả phát hiện xe thực tế. Mô hình nhận diện đồng thời các phương tiện trên làn đường với độ tin cậy ổn định, cung cấp tọa độ không gian chính xác cho thuật toán theo dõi kế tiếp."*

---

### SLIDE 9: BÁM ĐUÔI XE: KALMAN FILTER & IOU (Nguyên lý BoT-SORT)
- **Thời lượng**: 2:45 – 3:10 (25 giây).
- **Thao tác**: Chỉ 4 khối nội dung theo dõi trên màn hình.
- **Lời thoại**:
  > *"Để theo dõi liên tục phương tiện qua nhiều khung hình liên tiếp, hệ thống ứng dụng thuật toán **BoT-SORT** kết hợp bộ lọc Kalman Filter:*  
  > *• **Bộ lọc Kalman** ước lượng trạng thái chuyển động từ vị trí và vận tốc ở khung trước để dự đoán tọa độ của xe ở khung hiện tại.*  
  > *• Thuật toán tính toán ma trận tương quan không gian dựa trên chỉ số diện tích giao nhau **IoU**, kết hợp với điểm số tin cậy phát hiện (`fuse_score`).*  
  > *• Dựa trên ngưỡng liên kết `match_thresh: 0.8`, hệ thống so khớp hộp bao dự đoán với hộp bao phát hiện thực tế để duy trì Track ID ổn định cho từng phương tiện.*  
  > *Nhờ đó, hệ thống hạn chế được tình trạng đổi ID khi xe di chuyển qua góc quay camera."*

---

### SLIDE 10: BÁM ĐUÔI XE: KALMAN FILTER & IOU (Minh họa quỹ đạo bám đuôi)
- **Thời lượng**: 3:10 – 3:25 (15 giây).
- **Thao tác**: Chỉ vào hình ảnh quỹ đạo xe và hộp theo dõi trên slide.
- **Lời thoại**:
  > *"Hình ảnh trên slide minh họa vệt di chuyển của phương tiện qua thời gian. Nhờ cơ chế dự đoán chuyển động của Kalman Filter, hệ thống duy trì được quỹ đạo bám đuôi ổn định ngay cả trong các tình huống xe tạm thời bị che khuất một phần hoặc thay đổi vận tốc."*

---

### SLIDE 11: CẮT BIỂN SỐ & GHÉP VÀO ĐÚNG XE (Bước 2 - Hungarian Matching)
- **Thời lượng**: 3:25 – 3:50 (25 giây).
- **Thao tác**: Chỉ sang cột Định vị biển số ➔ Chỉ sang cột Ghép biển số bằng Hungarian.
- **Lời thoại**:
  > *"Sau khi theo dõi thân xe, chúng ta bước sang **Bước 2: Định vị biển số và ghép vào đúng xe**:*  
  > *• Mô hình YOLOv8 thứ hai được huấn luyện chuyên biệt để định vị vùng biển số xe.*  
  > *• Để giải quyết bài toán ghép đúng biển số vào từng phương tiện tương ứng, trong mã nguồn `src/engine.py`, nhóm triển khai giải thuật **Hungarian Matching** (`linear_sum_assignment`):*  
  > *Hệ thống xây dựng ma trận chi phí dựa trên điểm số kết hợp giữa mức độ bao phủ diện tích biển số (IoPA) và khoảng cách tâm tương đối, có điều kiện nới biên an toàn quanh hộp xe. Giải thuật này **giúp tối ưu hóa việc ghép cặp 1-1 và giảm thiểu đáng kể khả năng gán nhầm biển số giữa các xe di chuyển gần nhau**."*

---

### SLIDE 12: QUY TRÌNH XÂY DỰNG MÔ HÌNH YOLOV8
- **Thời lượng**: 3:50 – 4:05 (15 giây).
- **Thao tác**: Chỉ sơ đồ 4 giai đoạn trên slide.
- **Lời thoại**:
  > *"Để xây dựng mô hình định vị biển số đạt hiệu năng cao, nhóm tiến hành quy trình thực nghiệm qua 4 giai đoạn: Thu thập và chuẩn bị dữ liệu ➔ Thiết kế kiến trúc mạng ➔ Lựa chọn các hàm mất mát ➔ và Cấu hình lịch trình huấn luyện."*

---

### SLIDE 13: THU THẬP & GÁN NHÃN DỮ LIỆU
- **Thời lượng**: 4:05 – 4:30 (25 giây).
- **Thao tác**: Chỉ các con số chính: 8,000 ảnh ➔ 2,500 ảnh VN ➔ Tỷ lệ 80% Train / 20% Val.
- **Lời thoại**:
  > *"Về cơ cấu dữ liệu huấn luyện:*  
  > *• Nhóm áp dụng phương pháp Transfer Learning: Bắt đầu từ mô hình pre-train trên **8,000 ảnh phương tiện giao thông** để học các đặc trưng cạnh và hình học cơ bản.*  
  > *• Sau đó, nhóm tiến hành Fine-tune chuyên sâu trên tập dữ liệu hơn **2,500 ảnh biển số thực tế tại Việt Nam**, bao gồm cả biển vuông và biển dài dưới nhiều điều kiện ánh sáng và góc chụp khác nhau.*  
  > *• Toàn bộ tập dữ liệu được **phân chia theo tỷ lệ 80% Train và 20% Validation để đánh giá trên tập validation**.*  
  > *• Đồng thời, nhóm cũng chuẩn bị tập dữ liệu 15,120 mẫu ký tự cho 36 nhãn (0–9 và A–Z) với 350 ảnh train và 70 ảnh val cho mỗi lớp nhằm phục vụ nghiên cứu mô hình nhận diện ký tự."*

---

### SLIDE 14: KIẾN TRÚC MẠNG YOLOV8 (Backbone, Neck, Decoupled Head)
- **Thời lượng**: 4:30 – 5:00 (30 giây).
- **Thao tác**: Lia bút chỉ qua 3 khối: Backbone ➔ Neck (PAN-FPN) ➔ Decoupled Head.
- **Lời thoại**:
  > *"Về kiến trúc mạng YOLOv8 phát hiện biển số gồm 3 thành phần chính:*  
  > *• **Backbone:** Tận dụng khối **C2f** với cơ chế tách đôi nhánh gradient để trích xuất đặc trưng sâu hiệu quả, kết hợp tầng gom tụ không gian **SPPF** giúp giảm thiểu khối lượng tính toán.*  
  > *• **Neck (PAN-FPN):** Kết hợp mạng FPN truyền đặc trưng ngữ nghĩa từ trên xuống và mạng PAN truyền đặc trưng vị trí không gian từ dưới lên. Kiến trúc cung cấp 3 tầng phát hiện, trong đó tầng **P3 (kích thước $80 \times 80$)** đóng vai trò chủ đạo để phát hiện các biển số có kích thước nhỏ ở cự ly xa.*  
  > *• **Decoupled Head:** Tách riêng độc lập nhánh phân loại nhãn và nhánh hồi quy tọa độ, giúp quá trình lan truyền ngược gradient không bị xung đột giữa hai nhiệm vụ."*

---

### SLIDE 15: CÁC HÀM MẤT MÁT CỦA YOLOV8 (CIoU + DFL + BCE)
- **Thời lượng**: 5:00 – 5:30 (30 giây).
- **Thao tác**: Chỉ 3 thành phần công thức toán trên slide: $L_{CIoU}$, $L_{DFL}$, $L_{BCE}$.
- **Lời thoại**:
  > *"Để tối ưu hóa độ chính xác tọa độ hộp bao và hạn chế mờ nhòe cạnh viền, cấu hình huấn luyện thiết lập tổng hàm mất mát kết hợp 3 thành phần:*  
  > *• **Complete IoU Loss ($L_{CIoU}$ với trọng số 7.5):** Tối ưu đồng thời độ trùng lặp diện tích, khoảng cách giữa hai tâm và tỷ lệ kích thước hộp bao.*  
  > *• **Distribution Focal Loss ($L_{DFL}$ với trọng số 1.5):** Hồi quy tọa độ biên dưới dạng phân phối xác suất liên tục quanh giá trị thực, hỗ trợ định vị chính xác cạnh viền khi biển số bị bóng râm hoặc chói sáng.*  
  > *• **Binary Cross-Entropy Loss ($L_{BCE}$ với trọng số 0.5):** BCE là thành phần loss dùng để tối ưu nhánh phân loại nhãn đối tượng.*  
  > *Việc đặt trọng số $7.5$ cho Box Loss và $1.5$ cho DFL thể hiện rõ mục tiêu ưu tiên độ chính xác vị trí hộp bao để chuẩn bị vùng ảnh cắt tốt nhất cho công đoạn nhận diện ký tự."*

---

### SLIDE 16: LỊCH TRÌNH HUẤN LUYỆN & LEARNING RATE (Slide Chốt Hạ)
- **Thời lượng**: 5:30 – 6:00 (30 giây).
- **Thao tác**: Chỉ vào khung Diễn tiến log thực tế qua 30 Epochs ở giữa slide ➔ Chỉ các mốc Warmup và close_mosaic.
- **Lời thoại**:
  > *"Trên Slide 16 là diễn tiến huấn luyện thực tế qua **30 Epochs** được trích xuất từ tệp log `results.csv` của lần huấn luyện tạo ra checkpoint `best.pt`:*  
  > *• **Về Learning Rate:** Theo nhật ký thực nghiệm trong `results.csv`, dù cấu hình `args.yaml` đặt `warmup_epochs: 3.0`, giá trị learning rate ghi nhận ở cuối mỗi epoch tiếp tục tăng từ $0.000200$ lên mức đỉnh $0.001443$ tại Epoch 9, sau đó bước vào chu trình suy giảm tuyến tính (`cos_lr: false`) đều đặn về $8.6 \times 10^{-5}$ ở Epoch 30. Tại **Epoch 15**, mAP@50 đạt mức sớm $98.17\%$ và Recall đạt $97.72\%.*  
  > *• **Giai đoạn tắt Mosaic (Epoch 21 đến 30):** Cấu hình `close_mosaic: 10` tắt hoàn toàn kỹ thuật ghép ảnh nhân tạo trong 10 epoch cuối để mô hình tinh chỉnh trên phân phối ảnh thực tế.*  
  > *Đến giai đoạn cuối:*  
  > *• Chỉ số **mAP@50-95 đạt đỉnh toàn cục 75.15% ở Epoch 26 và 28**, sau đó duy trì ở mức 74.84% tại Epoch 30.*  
  > *• Tại **Epoch 30 nghiệm thu**, **Precision đạt 99.05% (xấp xỉ 99%)**, **mAP@50 đạt 98.17%**, và độ phủ **Recall đạt 87.50%** trên tập dữ liệu validation.*  
  > <br>
  > ***[CÂU CHUYỂN GIAO SANG BẠN 2]:***  
  > *Dựa trên lịch trình huấn luyện này, mô hình đã hội tụ với bộ trọng số nghiệm thu `best.pt`. Tiếp theo đây, em xin kính mời **bạn [Tên Bạn 2]** trình bày chi tiết bảng chỉ số nghiệm thu ở Slide 17, các biểu đồ đánh giá thực nghiệm ở Slide 18, 19 và các bước xử lý tiếp theo của đề tài.*  
  > ***Em xin chân thành cảm ơn Quý Thầy Cô đã lắng nghe phần trình bày của em ạ!"***
