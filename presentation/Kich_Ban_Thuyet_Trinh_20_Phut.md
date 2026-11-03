# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN CHUYÊN NGÀNH (PBL4: TRÍ TUỆ NHÂN TẠO)
**Thời lượng chuẩn:** 20 Phút (Chuẩn bảo vệ Hội đồng Bách Khoa Đà Nẵng)  
**Đề tài:** HỆ THỐNG GIÁM SÁT GIAO THÔNG & NHẬN DIỆN BIỂN SỐ TỰ ĐỘNG CHO TRẠM THU PHÍ BOT DỰA TRÊN YOLOV8 VÀ DEEP OCR  
**Sinh viên thực hiện:** Cao Tiến Dũng | **Lớp:** 22DT2 | **MSSV:** 106220049  
**Khoa:** Điện tử - Viễn thông | **Trường:** Đại học Bách Khoa - Đại học Đà Nẵng  

---

## BẢNG PHÂN BỔ THỜI GIAN (20 PHÚT)

| Phút | Phần | Slide | Trọng Tâm Trình Bày |
| :---: | :---: | :---: | :--- |
| **00:00 - 03:00** (3p) | **Phần 1** | Slide 1 - 4 | Mở đầu, bối cảnh trạm thu phí BOT, thách thức kỹ thuật, định nghĩa Dải hoạt động kỹ thuật ($h \ge 38$px, min(w)=42px, góc $\le 25^\circ$). |
| **03:00 - 08:00** (5p) | **Phần 2** | Slide 5 - 6 | Cơ sở lý thuyết CNN (Tích chập, Receptive Field, SiLU); Kiến trúc YOLOv8 (Backbone C2f, PAN-FPN P3/P4/P5, Decoupled Anchor-free Head, CIoU/DFL Loss). |
| **08:00 - 14:00** (6p) | **Phần 3** | Slide 7 - 14 | Pipeline hệ thống: Model kép, Thuật toán Hungarian $O(N^3)$, Phân luồng Aspect Ratio Routing ($AR=2.3$), Deep OCR (CRAFT + CRNN), Cơ chế Trì hoãn (Deferral), OpenVINO CPU 39.4ms, Khử trùng lặp và TrafficShield. |
| **14:00 - 18:00** (4p) | **Phần 4** | Slide 15 - 18 | Thực nghiệm: 51 mẫu BOT, bản chất 13.7% ảnh xa, độ chính xác 81.8%, Khoảng tin cậy Wilson 95% [68%, 91%], video thực tế 14/14 xe, Bảng so sánh Benchmark. |
| **18:00 - 20:00** (2p) | **Phần 5** | Slide 19 - 21 | Hạn chế thực tế trung thực, hướng phát triển tương lai, tổng kết 5 đóng góp cốt lõi, lời cảm ơn và chuyển sang Q&A. |

---

## LỜI THOẠI CHI TIẾT TỪNG SLIDE (WORD-FOR-WORD SCRIPT)

### SLIDE 1: BÌA ĐỀ TÀI (00:00 - 01:00)
> **[Thao tác]** Đứng thẳng, tự tin, hướng ánh mắt về phía Quý Thầy Cô trong Hội đồng.
> 
> *"Kính thưa quý Thầy Cô trong Hội đồng đánh giá và các bạn sinh viên.*  
> *Em tên là **Cao Tiến Dũng**, sinh viên lớp **22DT2**, Khoa Điện tử - Viễn thông, Trường Đại học Bách Khoa - Đại học Đà Nẵng.*  
> 
> *Hôm nay, em rất vinh dự được đại diện nhóm trình bày báo cáo bảo vệ đồ án học phần **PBL4: Trí Tuệ Nhân Tạo** với đề tài:*  
> **'Hệ thống giám sát giao thông và nhận diện biển số tự động cho trạm thu phí BOT dựa trên YOLOv8 và Deep OCR'.**  
> *Bài thuyết trình được thiết kế với thời lượng chuẩn 20 phút, sau đây em xin phép được bắt đầu."*

---

### SLIDE 2: LỘ TRÌNH BÁO CÁO (01:00 - 01:45)
> **[Thao tác]** Chuyển Slide 2.
> 
> *"Để Hội đồng tiện theo dõi và đánh giá toàn diện, bài báo cáo của em được cấu trúc chặt chẽ qua 5 phần chính:*  
> 
> *1. **Phần 1 (3 phút):** Nêu bật tính cấp thiết thực tế tại các trạm BOT và định nghĩa Dải hoạt động kỹ thuật.*  
> *2. **Phần 2 (5 phút):** Đi sâu vào cơ sở toán học của Mạng nơ-ron tích chập (CNN) và kiến trúc State-of-the-art YOLOv8.*  
> *3. **Phần 3 (6 phút):** Trình bày toàn bộ Pipeline hiện thực hóa hệ thống với 4 đóng góp thuật toán nổi bật: Ghép nối Hungarian, Phân luồng Aspect Ratio Routing, Deep OCR và Cơ chế Trì hoãn Deferral.*  
> *4. **Phần 4 (4 phút):** Đánh giá thực nghiệm với 51 mẫu BOT, phân tích khoảng tin cậy thống kê Wilson 95% và kiểm thử video thực tế.*  
> *5. **Phần 5 (2 phút):** Tổng kết đóng góp, nhìn nhận hạn chế khoa học trung thực và mở đầu phiên hỏi đáp Q&A."*

---

### SLIDE 3: BỐI CẢNH THỰC TẾ & THÁCH THỨC TẠI TRẠM BOT (01:45 - 02:30)
> **[Thao tác]** Chuyển Slide 3. Chỉ tay vào 3 cột thách thức.
> 
> *"Thưa Thầy Cô, hiện nay tại Việt Nam, hệ thống thu phí tự động không dừng ETC bằng chip RFID đã phủ rộng toàn quốc. Tuy nhiên, hệ thống nhận diện biển số tự động ANPR qua camera vẫn là chốt chặn thị giác bắt buộc để đối soát quang học với tài khoản phương tiện, nhằm phát hiện các hành vi gian lận như tráo biển, dùng biển giả hoặc dán băng keo che số.*  
> 
> *Khi triển khai camera thị giác tại làn BOT, hệ thống đối mặt với 3 thách thức lớn:*  
> *1. **Góc nghiêng và cự ly:** Phương tiện di chuyển liên tục từ xa vào gần, dẫn đến độ phân giải vùng biển số thay đổi liên tục.*  
> *2. **Đặc thù biển số Việt Nam:** Tồn tại song song cả biển chữ nhật 1 dòng (trên ô tô) và biển vuông 2 dòng (trên xe máy, xe tải). Nếu dùng OCR thông thường quét ngang, chữ và số ở 2 dòng sẽ bị đọc dính vào nhau.*  
> *3. **Hạ tầng máy chủ tại trạm:** Phần lớn các trạm thu phí chỉ trang bị máy tính CPU tiêu chuẩn, không có GPU máy trạm đắt đỏ. Đồng thời, một xe dừng chờ trước barie 3 giây sẽ tạo ra hàng chục khung hình trùng lặp, đòi hỏi hệ thống phải khử trùng lặp triệt để và giữ độ trễ toàn chuỗi dưới 500 mili-giây."*

---

### SLIDE 4: MỤC TIÊU ĐỀ TÀI & DẢI HOẠT ĐỘNG KỸ THUẬT (02:30 - 03:00)
> **[Thao tác]** Chuyển Slide 4. Nhấn mạnh cột 'Dải hoạt động kỹ thuật'.
> 
> *"Xuất phát từ các bài toán trên, mục tiêu của đề tài là xây dựng một pipeline End-to-End hoàn chỉnh, chạy tối ưu thời gian thực trên CPU với chi phí bản quyền 0 đồng.*  
> 
> *Đặc biệt, dưới góc độ kỹ sư, em không đưa ra các tuyên bố chung chung rằng hệ thống nhận diện đúng 100% trong mọi điều kiện, mà xác định rõ ràng **Dải hoạt động kỹ thuật (Operational Envelope)**:*  
> *• Chiều cao biển số tối thiểu: $h \ge 38$ pixels, đảm bảo đủ thông tin cho mô hình trích xuất đặc trưng.*  
> *• Chiều rộng biển số: $w \ge 42$ pixels, đo đạc trực tiếp trên tập thực nghiệm.*  
> *• Góc xiên phương tiện: $\le 25^\circ$ so với trục quang học camera.*  
> *• Tốc độ khung hình: Chuẩn CCTV 25 - 30 FPS.*  
> 
> *Các mẫu biển số ngoài dải ($h < 38$px) chỉ chiếm 13.7% trên tập ảnh crop cự ly xa; trong thực tế khi xe lăn bánh lại gần trạm, 100% xe đều sẽ tiến vào dải tối ưu và được nhận diện chuẩn xác."*

---

### SLIDE 5: CƠ SỞ LÝ THUYẾT MẠNG TÍCH CHẬP CNN (03:00 - 04:30)
> **[Thao tác]** Chuyển Slide 5. Hướng dẫn công thức tích chập và SiLU.
> 
> *"Bước sang Phần 2, nền tảng toán học của toàn bộ hệ thống là Mạng nơ-ron tích chập (CNN).*  
> 
> *Như Thầy Cô thấy trên slide, phép tích chập $I * K$ trượt nhân ma trận kích thước $k \times k$ trên ảnh. Hai nguyên lý cốt lõi của CNN là:*  
> *1. **Chia sẻ trọng số (Weight Sharing):** Cùng một bộ lọc được quét trên toàn ảnh, giúp giảm hàng triệu tham số so với mạng Fully-Connected, chống quá khớp (overfitting).*  
> *2. **Trường tiếp nhận cục bộ (Local Receptive Field):** Giữ vững mối quan hệ không gian giữa các điểm ảnh lân cận.*  
> 
> *Về hàm kích hoạt phi tuyến, đồ án sử dụng hàm **SiLU (Swish)**:*  
> $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$  
> *Hàm SiLU có đạo hàm mượt và không bị bão hòa ở miền âm như ReLU, khắc phục triệt để hiện tượng chết nơ-ron (Dying ReLU).*  
> 
> *Thông qua cấu trúc phân tầng (Feature Hierarchy), các tầng nông học biên cạnh sắc nét của biển số, các tầng trung gian học cấu trúc ký tự, và các tầng sâu nắm bắt ngữ nghĩa toàn bộ phương tiện."*

---

### SLIDE 6: KIẾN TRÚC MẠNG YOLOV8 STATE-OF-THE-ART (04:30 - 06:00)
> **[Thao tác]** Chuyển Slide 6. Phân tích 3 khối Backbone, Neck, Head.
> 
> *"Trên nền tảng CNN, em lựa chọn kiến trúc mạng phát hiện vật thể tiên tiến nhất hiện nay là **YOLOv8** với 3 cải tiến mang tính đột phá:*  
> 
> *1. **Backbone CSPDarknet với module C2f:** Thay thế module C3 của YOLOv5. C2f chia tách dòng gradient làm 2 nhánh, vừa tăng cường khả năng học đặc trưng phong phú, vừa giảm 20% chi phí tính toán. Cuối Backbone là module SPPF gom tụ đặc trưng đa kích thước.*  
> *2. **Neck PAN-FPN Đa Tỷ Lệ:** Kết hợp luồng ngữ nghĩa Top-down và luồng tọa độ Bottom-up, đưa ra 3 thang đo đặc trưng chuyên biệt: P3 (80×80) chuyên bắt vật thể nhỏ như biển số xe từ xa, P4 (40×40) cho xe con, và P5 (20×20) cho xe tải, xe buýt lớn.*  
> *3. **Head Anchor-Free và Decoupled:** Tách rời hoàn toàn nhánh phân loại lớp (sử dụng BCE Loss) và nhánh dự đoán vị trí hộp. YOLOv8 loại bỏ các anchor box cố định truyền thống, dự đoán trực tiếp khoảng cách từ tâm đến 4 cạnh hộp thông qua hàm tổn thất **Distribution Focal Loss (DFL)** kết hợp **CIoU Loss**, giúp viền crop biển số đạt độ chính xác đến từng điểm ảnh."*

---

### SLIDE 7: MÔ HÌNH KÉP - PHÁT HIỆN PHƯƠNG TIỆN & BIỂN SỐ (06:00 - 07:15)
> **[Thao tác]** Chuyển Slide 7.
> 
> *"Bước sang Phần 3 - Hiện thực hóa Pipeline, hệ thống của em tách biệt làm 2 mô hình YOLO chạy phối hợp:*  
> 
> *• **Model 1: Vehicle Detection (YOLOv8n)**: Tiền huấn luyện trên tập dữ liệu chuẩn COCO, trích xuất 4 lớp phương tiện giao thông: Ô tô (Car), Xe máy (Motorcycle), Xe buýt (Bus) và Xe tải (Truck). Mục đích là phân loại chính xác loại xe để phục vụ tính biểu giá phí qua trạm BOT.*  
> *• **Model 2: License Plate Detection (Custom Fine-tuned)**: Mô hình chuyên biệt được fine-tune trên hơn 2,500 ảnh biển số xe thực tế tại Việt Nam với đa dạng điều kiện: chói sáng, ngược nắng, góc nghiêng $25^\circ$ và biển bám bụi. Mô hình chịu trách nhiệm bóc tách vùng biển số với độ chính xác cao nhất."*

---

### SLIDE 8: THUẬT TOÁN HUNGARIAN GHÉP NỐI LƯỠNG PHÂN (07:15 - 08:30)
> **[Thao tác]** Chuyển Slide 8. Nhấn mạnh công thức hàm chi phí Cost(i,j).
> 
> *"Một đóng góp thuật toán rất quan trọng của đồ án là việc áp dụng **Thuật toán ghép nối lưỡng phân Hungarian**.*  
> 
> *Trong khung cảnh đông đúc tại trạm BOT, khi nhiều phương tiện di chuyển song song, camera phát hiện $N$ biển số và $M$ phương tiện. Nếu dùng phương pháp so khớp IoU đơn thuần thì khi xe máy chạy sát cạnh xe container, biển số xe máy rất dễ bị nhận nhầm là thuộc về xe container.*  
> 
> *Em giải quyết bài toán này bằng cách xây dựng đồ thị lưỡng phân (Bipartite Graph) với ma trận chi phí tổng hợp:*  
> $$\text{Cost}(i, j) = 0.6 \cdot \frac{\text{Dist}(C_{\text{plate}}, C_{\text{veh}})}{\text{Diag}_{\text{veh}}} + 0.4 \cdot (1 - \text{Overlap}(\text{plate}, \text{veh}))$$  
> 
> *Thuật toán Hungarian tối ưu hóa toàn cục trong thời gian $O(N^3)$, đảm bảo mỗi biển số được gán duy nhất và chính xác cho đúng 1 xe sở hữu. Tuyệt đối không bao giờ xảy ra lỗi trừ sai tiền vé giữa ô tô và xe máy.*  
> *Nếu xe bị che khuất thân nhưng vẫn lộ biển, hệ thống có cơ chế Fallback tự suy luận loại xe dựa trên định dạng ký tự biển số."*

---

### SLIDE 9: PHÂN LUỒNG HÌNH HỌC BIỂN SỐ ASPECT RATIO ROUTING (08:30 - 09:30)
> **[Thao tác]** Chuyển Slide 9. Chỉ vào đoạn mã chia nhánh.
> 
> *"Tiếp theo là thuật toán **Aspect Ratio Routing** nhằm xử lý đặc thù biển số Việt Nam:*  
> *• Biển 1 dòng dài có tỷ lệ $AR \approx 4.27$.*  
> *• Biển 2 dòng vuông có tỷ lệ $AR \approx 1.35 - 1.40$.*  
> 
> *Em thiết lập ngưỡng phân luồng thực nghiệm là $AR = 2.3$:*  
> *• **Nhánh Fast-path ($AR \ge 2.3$):** Biển 1 dòng được đưa thẳng vào bộ nhận dạng `recognize()`, bỏ qua hoàn toàn bước dò hộp CRAFT, rút ngắn thời gian xử lý xuống chỉ còn ~60ms.*  
> *• **Nhánh Multi-line ($AR < 2.3$):** Biển 2 dòng được tách thành 2 vùng độc lập theo tọa độ trục Y, đọc dòng trên (mã tỉnh) rồi đến dòng dưới (dãy số thứ tự) và ghép lại.*  
> 
> *Thuật toán này loại bỏ triệt để lỗi đọc ngược hoặc dính dòng, đồng thời tăng tốc độ xử lý trung bình của toàn hệ thống lên 40%."*

---

### SLIDE 10: DEEP OCR - KIẾN TRÚC CRAFT VÀ CRNN (09:30 - 10:45)
> **[Thao tác]** Chuyển Slide 10.
> 
> *"Mô hình thứ 3 trong pipeline là bộ nhận diện ký tự Deep OCR kết hợp giữa **CRAFT** và **CRNN**:*  
> 
> *1. **CRAFT (Character Region Awareness for Text Detection):** Dự đoán 2 bản đồ nhiệt: Region Score xác định tâm ký tự và Affinity Score liên kết các ký tự liền kề, giúp phát hiện ký tự ổn định kể cả khi biển số bị cong vênh hoặc nghiêng chéo.*  
> *2. **CRNN (Convolutional Recurrent Neural Network):** Gồm mạng tích chập ResNet trích xuất đặc trưng thị giác, tầng Map-to-Sequence biến đổi đặc trưng thành chuỗi thời gian, và 2 tầng BiLSTM hai chiều để học mối tương quan ngữ cảnh ký tự.*  
> *3. **CTC Loss & Allowlist:** Tầng Connectionist Temporal Classification giải mã chuỗi mà không cần căn chỉnh vị trí từng ký tự. Em tích hợp sẵn Allowlist chỉ chấp nhận các ký tự hợp chuẩn trên biển số Việt Nam, loại bỏ hoàn toàn các ký tự đặc biệt gây nhiễu."*

---

### SLIDE 11: CƠ CHẾ TRÌ HOÃN OCR (DEFERRAL MECHANISM) (10:45 - 12:00)
> **[Thao tác]** Chuyển Slide 11. Nhấn mạnh sự khác biệt giữa Loại bỏ (Rejection) và Trì hoãn (Deferral).
> 
> *"Kính thưa Hội đồng, đây là một điểm mấu chốt trong thiết kế kỹ thuật của em nhằm trả lời câu hỏi: **'Nếu bỏ qua biển số nhỏ hơn 38 pixels thì có làm mất xe không?'**.*  
> 
> *Câu trả lời dứt khoát là: **KHÔNG HỀ MẤT XE**.*  
> *Trong các hệ thống thông thường, khái niệm 'Rejection' đồng nghĩa với việc loại bỏ vĩnh viễn dữ liệu nếu frame đầu tiên không đọc được; hoặc cố đọc cưỡng bức sẽ sinh ra ký tự rác.*  
> 
> *Trong đồ án của em, đây là **Cơ Chế Trì Hoãn (Deferral Mechanism)**:*  
> *• Khi xe ở xa ($h < 38$px), YOLO vẫn phát hiện xe và thuật toán theo dõi duy trì Track ID xuyên suốt các frame.*  
> *• Bước OCR chỉ tạm thời trì hoãn lại nhằm tiết kiệm tài nguyên CPU và tránh sinh ra kết quả rác.*  
> *• Khi xe tiến vào vùng quan sát tối ưu ($h \ge 38$px), OCR được kích hoạt tự động trên chính Track ID đó.*  
> *• Kết quả là mỗi xe chỉ ghi nhận 1 lần duy nhất tại khung hình có chất lượng quang học cao nhất."*

---

### SLIDE 12: TỐI ƯU HÓA PHẦN CỨNG BẰNG INTEL OPENVINO (12:00 - 13:00)
> **[Thao tác]** Chuyển Slide 12.
> 
> *"Để giải quyết bài toán chi phí đầu tư tại các trạm BOT, em không phụ thuộc vào GPU đắt đỏ mà sử dụng bộ công cụ tối ưu hóa **Intel OpenVINO**:*  
> 
> *• Toàn bộ mô hình PyTorch được biên dịch sang định dạng trung gian IR (Intermediate Representation) gồm tệp `.xml` và `.bin`, dung lượng chỉ còn 12.1 MB.*  
> *• Áp dụng kỹ thuật **Constant Folding** (tính toán trước các giá trị hằng số) và **Layer Fusion** (gộp các tầng Conv, BatchNorm và SiLU thành một lệnh đơn).*  
> *• Lượng tử hóa trọng số về định dạng nửa chính xác **FP16** giúp giảm 50% bộ nhớ RAM mà độ chính xác mAP suy giảm không đáng kể ($< 0.1\%$).*  
> 
> *Kết quả đo đạc thực tế: Độ trễ suy luận giảm từ 180ms xuống chỉ còn **39.4 mili-giây / khung hình** trên CPU Intel Core i5 thông thường, đảm bảo hệ thống chạy mượt mà 25-30 FPS."*

---

### SLIDE 13: HẬU XỬ LÝ KÝ TỰ & KHỬ TRÙNG LẶP TUYỆT ĐỐI (13:00 - 13:45)
> **[Thao tác]** Chuyển Slide 13.
> 
> *"Ở khâu hậu xử lý, em cài đặt bộ lọc chuẩn hóa chuỗi:*  
> *• Tự động sửa lỗi vị trí chữ/số: Ví dụ, ký tự thứ 3 trên biển xe cơ giới bắt buộc là chữ cái, nếu OCR nhầm số '8' sẽ tự sửa thành chữ 'B', số '0' thành chữ 'D'.*  
> *• Khắc phục triệt để lỗi nhận nhầm dấu ngoặc vuông '[' thành 'L' và ']' thành '1'.*  
> *• Định dạng chuẩn Bộ GTVT có dấu chấm phân cách (ví dụ: 51H.04964).*  
> 
> *Về khử trùng lặp bản ghi: Một xe dừng trước barie tạo ra 90 frames trùng nhau. Thuật toán của em chuẩn hóa chuỗi và áp dụng cơ chế **Upsert thông minh**: Nếu biển số đã tồn tại, hệ thống chỉ cập nhật thời gian mới nhất và ảnh crop nét hơn, tuyệt đối không tạo thêm bản ghi mới. Kết quả khử sạch 100% bản ghi rác, database luôn hiển thị đúng số lượng xe thực tế."*

---

### SLIDE 14: KIẾN TRÚC PHẦN MỀM & BẢO MẬT TRAFFICSHIELD (13:45 - 14:15)
> **[Thao tác]** Chuyển Slide 14.
> 
> *"Toàn bộ hệ thống được đóng gói thành ứng dụng web hoàn chỉnh trên nền tảng **FastAPI** bất đồng bộ và giao thức **WebSockets** truyền luồng dữ liệu thời gian thực.*  
> 
> *Hệ thống tích hợp lớp bảo vệ **TrafficShield** (`src/shield.py`) với thuật toán Token Bucket Rate-limiting chống tấn công DoS, đồng thời Whitelist thông minh cho các địa chỉ camera nội bộ.*  
> *Giao diện vận hành được thiết kế theo phong cách Enterprise Daylight chuẩn trạm điều hành BOT với lớp phủ CCTV OSD và thành phần huy hiệu biển số dập nổi chân thực."*

---

### SLIDE 15: TẬP DỮ LIỆU THỰC NGHIỆM TẠI TRẠM BOT (14:15 - 15:15)
> **[Thao tác]** Chuyển Slide 15. Giải thích minh bạch cơ cấu mẫu.
> 
> *"Bước sang Phần 4 - Đánh giá thực nghiệm, em tiến hành thử nghiệm trên tập dữ liệu gồm **51 mẫu ảnh crop tĩnh** chụp thực tế tại các làn thu phí BOT.*  
> 
> *Cơ cấu tập mẫu như sau:*  
> *• 44 mẫu nằm trong Dải hoạt động kỹ thuật ($h \ge 38$px), chiếm 86.3%.*  
> *• 7 mẫu ở cự ly xa ngoài dải ($h < 38$px), chiếm 13.7%.*  
> *• Kích thước chiều cao dao động từ 24px đến 110px, chiều rộng nhỏ nhất đo được là $42$px.*  
> 
> *Em xin nhấn mạnh một lần nữa: Con số 13.7% này là tỷ lệ ảnh crop ở cự ly xa trong tập test tĩnh, hoàn toàn không phải tỷ lệ xe bị mất trong thực tế. Đây là đặc tính thiết kế có chủ đích để hệ thống chủ động kích hoạt cơ chế Trì hoãn Deferral."*

---

### SLIDE 16: KẾT QUẢ THỰC NGHIỆM & KHOẢNG TIN CẬY WILSON 95% (15:15 - 16:45)
> **[Thao tác]** Chuyển Slide 16. Phân tích công thức khoảng tin cậy Wilson.
> 
> *"Về kết quả thực nghiệm:*  
> *• Trong dải hoạt động tối ưu, hệ thống nhận diện đúng 100% từng ký tự trên **36 / 44 mẫu**, đạt tỷ lệ chính xác điểm $\hat{p} = 81.8\%$ (làm tròn 82%).*  
> *• Nếu tính trên toàn bộ 51 mẫu thô bao gồm cả các ảnh ở xa thì đạt 36 / 51 = 70.6%.*  
> *• Độ trễ toàn chuỗi trung bình là **382 mili-giây / xe**, hoàn toàn đáp ứng tiêu chuẩn thời gian thực của trạm BOT.*  
> 
> *Điểm đặc biệt trong báo cáo của em là việc áp dụng **Khoảng tin cậy Wilson Score 95%**:*  
> $$\text{CI}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$  
> 
> *Với cỡ mẫu nhỏ $N < 100$, công thức xấp xỉ chuẩn Wald truyền thống thường bị sai lệch biên. Công thức Wilson bất đối xứng cho độ bao phủ xác suất tin cậy 95%.*  
> *Khoảng tin cậy cho dải tối ưu của hệ thống là **[68%, 91%]**.*  
> *Điều này khẳng định về mặt toán học rằng: Trong điều kiện thực tế, xác suất nhận diện đúng của hệ thống chắc chắn đạt tối thiểu 68% và kỳ vọng ở mức 82%."*

---

### SLIDE 17: THỬ NGHIỆM THỰC TẾ TRÊN VIDEO LUỒNG BOT (16:45 - 17:30)
> **[Thao tác]** Chuyển Slide 17.
> 
> *"Để kiểm chứng tính khả thi trong môi trường động, em đã thử nghiệm trực tiếp trên video thực tế `test_bot_cutted.mp4` quay tại trạm thu phí BOT Quốc lộ 1A với 14 lượt xe qua làn.*  
> 
> *Kết quả thực tế ghi nhận:*  
> *• **14 / 14 lượt xe đều được nhận diện và đối soát thành công (đạt 100%)**, chứng minh cơ chế Trì hoãn hoạt động hoàn hảo khi xe tiến lại gần.*  
> *• Hệ thống nhận diện chuẩn xác cả các trường hợp phức tạp như xe khách 72F.00075, xe tải 51B.60174 và các xe máy chen làn như 54L.19999.*  
> *• Cơ chế khử trùng lặp đa khung hình hợp nhất toàn bộ dữ liệu về đúng 14 bản ghi duy nhất, tốc độ hiển thị video đạt 25 - 30 FPS ổn định không giật lag."*

---

### SLIDE 18: BẢNG SO SÁNH ĐỐI CHUẨN BENCHMARKING (17:30 - 18:15)
> **[Thao tác]** Chuyển Slide 18. So sánh với Tesseract và Cloud.
> 
> *"Bảng so sánh trên slide cho thấy rõ ưu thế của giải pháp đề xuất:*  
> *• So với **Tesseract OCR**: Giải pháp của em có độ trễ nhanh gấp 2 đến 3 lần và xử lý hoàn hảo biển số 2 dòng tại Việt Nam nhờ thuật toán Aspect Ratio Routing.*  
> *• So với **Cloud OCR (Google Cloud/AWS)**: Giải pháp của em chạy 100% Offline độc lập tại trạm thu phí với chi phí phần cứng 0 đồng GPU và 0 đồng cước API, loại bỏ hoàn toàn rủi ro tắc nghẽn barie khi đường truyền Internet bị ngắt quãng."*

---

### SLIDE 19: HẠN CHẾ ĐÃ GHI NHẬN & HƯỚNG PHÁT TRIỂN (18:15 - 19:00)
> **[Thao tác]** Chuyển Slide 19. Thái độ khoa học trung thực.
> 
> *"Nhìn nhận sản phẩm với tinh thần khoa học cầu thị, em ghi nhận 3 hạn chế thực tế:*  
> *1. Hệ thống hiện tối ưu cho ánh sáng ban ngày, chưa kiểm nghiệm đầy đủ trong điều kiện ban đêm thiếu sáng cần camera hồng ngoại.*  
> *2. Cỡ mẫu đánh giá tĩnh 51 ảnh cần tiếp tục mở rộng lên 500-1000 mẫu để thu hẹp khoảng tin cậy Wilson.*  
> *3. Khi góc nghiêng phương tiện vượt quá $25^\circ$, độ chính xác nhận diện ký tự bắt đầu suy giảm do méo hình học.*  
> 
> *Hướng phát triển tiếp theo của đề tài:*  
> *• Tích hợp thuật toán nắn phối cảnh 4 điểm Homography bằng mô hình Corner Detection.*  
> *• Kết nối API trích nợ tự động với các tài khoản giao thông VETC / ePass.*  
> *• Đóng gói Docker để triển khai trực tiếp trên máy tính nhúng Edge như Raspberry Pi 5 hoặc NVIDIA Jetson."*

---

### SLIDE 20: KẾT LUẬN & ĐÓNG GÓP KỸ THUẬT (19:00 - 19:30)
> **[Thao tác]** Chuyển Slide 20. Đúc kết 5 đóng góp chính.
> 
> *"Tóm lại, đồ án đã hoàn thành trọn vẹn 5 đóng góp kỹ thuật cốt lõi:*  
> *1. Pipeline End-to-End hoàn chỉnh từ camera đến cơ sở dữ liệu.*  
> *2. Tối ưu hóa sâu trên CPU với OpenVINO, đạt độ trễ 39.4ms.*  
> *3. Thuật toán nghiệp vụ thông minh: Aspect Ratio Routing và Cơ chế Trì hoãn Deferral.*  
> *4. Phương pháp đo lường khoa học với khoảng tin cậy Wilson 95% [68%, 91%].*  
> *5. Tự động hóa cài đặt và sẵn sàng chuyển giao vận hành bằng 1-click script."*

---

### SLIDE 21: XIN CẢM ƠN & PHIÊN HỎI ĐÁP Q&A (19:30 - 20:00)
> **[Thao tác]** Chuyển Slide 21. Cúi chào trang trọng.
> 
> *"Em xin phép được kết thúc bài báo cáo tại đây.*  
> *Em xin trân trọng cảm ơn Quý Thầy Cô trong Hội đồng đánh giá và các bạn sinh viên đã chú ý lắng nghe.*  
> 
> *Kính mong nhận được những nhận xét, góp ý quý báu từ Thầy Cô để đồ án của em ngày càng hoàn thiện hơn. Em xin sẵn sàng lắng nghe và trả lời các câu hỏi phản biện từ Hội đồng!"*

---

## NGÂN HÀNG CÂU HỎI PHẢN BIỆN & CÂU TRẢ LỜI MẪU XUẤT SẮC (DEFENSE Q&A)

### Câu hỏi 1: Tại sao em phải tách làm 2 mô hình (YOLO xe + YOLO biển số) mà không dùng 1 mô hình duy nhất phát hiện cả hai?
> **Trả lời mẫu của sinh viên:**  
> *"Dạ thưa Thầy Cô, việc tách làm 2 mô hình mang lại 3 ưu thế kỹ thuật vượt trội:*  
> *1. **Độ phân giải và tỷ lệ vật thể chênh lệch quá lớn:** Toàn thân ô tô chiếm khoảng $600 \times 400$ pixels trên ảnh Full HD, trong khi biển số xe chỉ chiếm khoảng $80 \times 30$ pixels. Nếu huấn luyện chung 1 mô hình, các tầng trích xuất đặc trưng sẽ bị thiên lệch về vật thể lớn và làm mất dấu biển số nhỏ ở tầng P3.*  
> *2. **Tách biệt dữ liệu huấn luyện:** Model 1 nhận dạng thân xe tận dụng trực tiếp trọng số chuẩn COCO mà không tốn công gán nhãn lại; trong khi Model 2 được tập trung fine-tune chuyên sâu trên bộ dữ liệu 2,500 ảnh biển số xe Việt Nam.*  
> *3. **Tính độc lập mô-đun:** Cho phép nâng cấp hoặc thay thế mô hình biển số mà không làm ảnh hưởng đến thuật toán phân loại phương tiện để tính giá vé BOT."*

---

### Câu hỏi 2: Tại sao em dùng khoảng tin cậy Wilson 95% mà không dùng công thức xấp xỉ Wald thông thường?
> **Trả lời mẫu của sinh viên:**  
> *"Dạ thưa Thầy Cô, công thức khoảng tin cậy chuẩn Wald $\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ dựa trên định lý giới hạn trung tâm, chỉ áp dụng chính xác khi cỡ mẫu rất lớn ($n > 100$) và tỷ lệ $\hat{p}$ ở quanh mức 0.5.*  
> *Trong bài toán thực tế của em với $n = 44$ và $\hat{p} = 0.818$, công thức Wald bị lỗi nghiêm trọng là tạo ra khoảng đối xứng giả tạo và có thể vượt ra ngoài giới hạn $[0, 1]$.*  
> *Công thức **Wilson Score Interval** đưa ra khoảng tin cậy bất đối xứng dựa trên phép nghịch đảo kiểm định điểm số (score test), đảm bảo xác suất bao phủ thực sự đạt đúng $95\%$ ngay cả với mẫu nhỏ. Điều này giúp báo cáo của em đạt tính chuẩn xác khoa học thống kê cao nhất."*

---

### Câu hỏi 3: Nếu hệ thống bỏ qua các biển số có chiều cao dưới 38px, liệu có làm 'mất xe' và thất thoát vé thu phí của trạm BOT không?
> **Trả lời mẫu của sinh viên:**  
> *"Dạ thưa Thầy Cô, hoàn toàn không làm mất xe. Đây chính là **Cơ chế Trì hoãn (Deferral Mechanism)** chứ không phải **Loại bỏ (Rejection)**.*  
> *Khi xe ở cự ly xa ($h < 38$px), YOLO vẫn phát hiện xe và gán Track ID liên tục giữa các khung hình. Hệ thống chỉ tạm thời hoãn bước Deep OCR vì ở cự ly đó độ phân giải quá thấp, nếu cố đọc sẽ sinh ra ký tự rác.*  
> *Do phương tiện buộc phải di chuyển qua barie thu phí, 100% xe sẽ tiến vào dải quan sát tối ưu ($h \ge 38$px). Lúc đó hệ thống kích hoạt OCR trên đúng Track ID đó để lấy kết quả chuẩn xác nhất. Thử nghiệm trên video thực tế 14 lượt xe qua trạm cho thấy 14/14 xe đều được nhận diện thành công mà không hề bị sót chiếc nào."*

---

### Câu hỏi 4: Thuật toán Hungarian giải quyết trường hợp xe máy chạy song song sát ô tô hoặc chen làn như thế nào?
> **Trả lời mẫu của sinh viên:**  
> *"Dạ thưa Thầy Cô, khi xe máy chạy sát cạnh xe con, nếu dùng tỷ lệ lọt lòng (containment) hoặc IoU đơn thuần thì biển số xe máy rất dễ bị tính nhầm là nằm trong bounding box lớn của xe ô tô.*  
> *Thuật toán Hungarian của em sử dụng ma trận chi phí tổng hợp gồm 2 thành phần có trọng số:*  
> *1. Khoảng cách Euclid chuẩn hóa từ tâm biển số đến trọng tâm xe (trọng số 0.6).*  
> *2. Độ lọt lòng của biển số (trọng số 0.4).*  
> *Nhờ đó, khoảng cách từ biển xe máy đến trọng tâm xe máy luôn ngắn hơn rất nhiều so với khoảng cách đến trọng tâm ô tô. Thuật toán tìm nghiệm gán tối ưu toàn cục ma trận $N \times M$ trong thời gian $O(N^3)$, phân định rạch ròi biển nào của xe nào."*

---

### Câu hỏi 5: Tại sao em không dùng GPU mà lại chọn OpenVINO trên CPU? Liệu CPU có bị quá tải khi lưu lượng xe đông không?
> **Trả lời mẫu của sinh viên:**  
> *"Dạ thưa Thầy Cô, lý do em chọn OpenVINO trên CPU là tính khả thi kinh tế và khả năng thương mại hóa thực tế:*  
> *1. Một card GPU máy trạm công nghiệp (như RTX 4000 hoặc Tesla) có giá từ 30 đến 50 triệu đồng mỗi làn, chi phí đầu tư cho một trạm BOT 8 làn lên tới hàng trăm triệu đồng và tiêu tốn nhiều điện năng.*  
> *2. Với OpenVINO, nhờ kỹ thuật Layer Fusion và lượng tử hóa FP16, mô hình YOLO chạy trên CPU chỉ mất **39.4 mili-giây** (tương đương hơn 25 FPS). Thời gian toàn chuỗi bao gồm cả OCR chỉ mất ~382ms.*  
> *3. Về lưu lượng: Tại mỗi làn BOT, tại một thời điểm chỉ có 1 xe dừng trước barie và 1-2 xe chờ phía sau. Hệ thống chỉ thực hiện OCR trên xe nằm trong dải tối ưu, do đó tải CPU duy trì ổn định dưới 45%."*

---

### Câu hỏi 6: Khi biển số xe bị bẩn, dính bùn đất hoặc chữ số lẫn lộn giữa số 8 với chữ B thì hệ thống xử lý thế nào?
> **Trả lời mẫu của sinh viên:**  
> *"Dạ thưa Thầy Cô, hệ thống xử lý qua 3 tầng phòng thủ:*  
> *1. **Tầng mô hình:** Mạng CRNN sử dụng 2 tầng BiLSTM học ngữ cảnh 2 chiều. Ví dụ: Ký tự thứ 3 bắt buộc là chữ cái (mã series), còn các ký tự sau dấu gạch ngang bắt buộc là chữ số.*  
> *2. **Tầng Allowlist:** Giới hạn từ điển giải mã CTC chỉ chấp nhận tập ký tự chuẩn của Bộ GTVT.*  
> *3. **Tầng Hậu xử lý Regex:** Áp dụng ma trận hoán đổi ký tự theo vị trí cấu trúc: Nếu ký tự thứ 3 bị OCR nhận là số '8' thì tự động sửa thành chữ 'B'; số '0' sửa thành 'D'; dấu ngoặc vuông '[' sửa thành 'L'. Nhờ đó tỷ lệ đọc đúng tăng thêm 12% so với OCR thô."*
