# Architecture alternative — architecture

Bản riêng theo yêu cầu 2026-09-14: **AWS-style architecture + network topology**.
Theo yêu cầu bổ sung: hiển thị trong khu vực architecture riêng, thêm demo ngay dưới project có video và cải thiện frontend. Không sửa Projects.astro hay CSS global cũ.

`src/pages/index.astro`: `USE_PORTFOLIO_FRONTEND = true` chọn frontend mới (`PortfolioLayout` + `Portfolio`); đặt `false` để trở lại frontend cũ. Giữ `SHOW_EXISTING_PROJECTS = false` theo lựa chọn của chủ portfolio; `SHOW_ARCHITECTURE` điều khiển atlas. Khi dùng frontend cũ, bản architecture có nút ẩn/hiện native details. Component độc lập: `src/components/Architecture.astro`.

## Xem và chỉnh sửa

- Preview độc lập: `public/architecture/index.html` (mở trực tiếp trong trình duyệt).
- Nếu portfolio dev server đang chạy: `/architecture/index.html`.
- `public/architecture/portfolio-architecture.drawio`: 6 trang, chỉnh sửa bằng diagrams.net hoặc ứng dụng draw.io. Double-click nhãn để sửa, kéo node để đổi vị trí.
- 6 file SVG bên cạnh HTML để nhúng vào portfolio bằng `<img>` hoặc tải riêng.
- `render.mjs`: nguồn hình học chung cho SVG và draw.io; chỉ in output, không tự ghi đè file.

## Design decisions

Dùng workflow của [GitHub Awesome Copilot — draw-io-diagram-generator](https://github.com/github/awesome-copilot/blob/main/skills/draw-io-diagram-generator/SKILL.md): mxGraph XML, parent/child boundaries, orthogonal connectors, editable multipage file, validation of IDs and endpoints. Tìm skill bằng `find-skills`; `frontend-design` hướng dẫn trang preview nhẹ và typography.

Tham chiếu cách trình bày [AWS Architecture Icons](https://aws.amazon.com/architecture/icons/). Logo thật cho Oracle, Hadoop, Spark, ZooKeeper, PostgreSQL, Debezium, Kafka, Confluent, Iceberg, Rust và NixOS; xem `public/architecture/icons/ATTRIBUTION.md`. Biểu tượng tự vẽ chỉ cho hạ tầng (server, switch, firewall) và chức năng nội bộ (memory, log, file). Không gán EC2, VPC, S3, RDS hoặc Availability Zone khi source không có. SVG nhúng logo để tải về không phụ thuộc URL bên ngoài.

## Evidence đọc trực tiếp

### Network + Oracle RAC

- `/home/tandat/Desktop/de_lab/README.md`: bốn CIDR sau migration, VM IP/VIP, SCAN .30, standby .40, allowlist 1521 cho SCAN + VIP.
- `/home/tandat/Desktop/de_lab/PLAN-FIREWALL-L4-SEGREGATION.md`: runbook triển khai firewall (không chạy script/VM để xác minh trạng thái live).
- `/home/tandat/Desktop/tandat_homelab/oracle_rac/STEP-12-RAC-DATAGUARD-STANDBY.md`: physical standby, redo transport, Broker, Observer.
- `/home/tandat/Desktop/tandat_homelab/oracle_rac/STEP-16-RAC-FSFO-TROUBLESHOOTING.md`: output Broker ghi MaxPerformance; tránh suy diễn SLA zero data loss.
- [Public Oracle repo](https://github.com/tandat8896/oracle-rac-data-guard-lab): cross-check cấu trúc RAC hai node, ASM, standby trên một NixOS host.

Lấy IP hiện tại từ de_lab thay vì copy IP cũ trong runbook RAC. SCAN thực hiện redirect tới local listener qua VIP. ASM là storage chia sẻ, không phải thành phần vận chuyển redo. Observer đặt ở vị trí logic, không khẳng định một VM độc lập hay site DR. Topology tổng quan lược bỏ cluster internal flows và shared disks vì đã có trang Oracle chi tiết.

### Hadoop + Spark

### Oracle memory / processes

- `/home/tandat/Desktop/cv/DBA/docs/query-tuning/docs/memory-tuning/01_sga_tuning.md` và `02_pga_tuning.md`: output trước tuning có SGA_TARGET 1610612736, MEMORY_TARGET 0, PGA_AGGREGATE_TARGET 536870912 và LIMIT 2147483648. Các snippet 2G/4G sau đó là đề xuất tuning, không dùng để khẳng định trạng thái live.
- `13_buffer_cache_flow.md` trong tài liệu query tuning: buffer cache, redo và commit. Demo `-eH738r5fRQ` lấy từ README Oracle của chủ portfolio; không gán demo này cho project khác.
- [Oracle process architecture](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/process-architecture.html) và [memory architecture](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/memory-architecture.html): đối chiếu LGWR, DBWn, CKPT, SGA/PGA. COMMIT ACK minh họa default WAIT, không đợi DBWn flush datafiles.

### Hadoop + Spark placement

- `/home/tandat/Desktop/de_lab/README-BIGDATA-INFRA-HARDENING.md`: phần HA cuối tài liệu xác nhận NameNode standby, Spark master standby trên workernode1 và logical nameservice.
- `/home/tandat/Desktop/de_lab/PLAN-HA-DR-BIGDATA.md`: ZooKeeper / JournalNode trên ba VM, nhánh Spark Standalone recovery, nameservice mycluster, ZKFC.

Tài liệu chứa cả trạng thái trước HA và kế hoạch; hình chỉ mô tả topology cấu hình được tài liệu HA hỗ trợ, không tự công bố benchmark. DataNode thể hiện trên hai worker theo bảng dịch vụ; runbook restart còn nhắc cả datanode nên không khẳng định pool chỉ có đúng hai DataNode đang live. Cần `jps`/config live nếu muốn chốt placement đầy đủ. Sơ đồ cố ý lược YARN và History Server để tập trung HDFS/Spark HA. Node active/standby không phải health check hiện tại.

### CDC / Iceberg

- `/home/tandat/Desktop/cv/DE/migration-data-pipeline/README.md`: local PostgreSQL pgoutput → Debezium → Kafka; Registry dùng BACKWARD.
- `iceberg/scripts/10_v4_confluent_avro_to_iceberg.scala` trong repo đó: `spark.read.format("kafka")`, schema lookup, chỉ giữ latest schema ID, `createOrReplace` Bronze. Cuối script ghi streaming job là next step.
- `iceberg/scripts/12_scd_type2.scala`: batch Bronze → Silver; chỉ lọc r/u/c, bỏ d, version theo descending rank, valid_to chưa đóng interval bằng sự kiện kế tiếp.

Vì vậy nhãn ghi batch load và Silver history exercise, không tuyên bố Structured Streaming/SCD2 hoàn chỉnh. Registry đặt ở side-channel metadata. Không thêm S3, MinIO, managed Kafka hoặc target Supabase khi đang vẽ implemented path này.

### tinydb

- `/home/tandat/Desktop/tinydb/README.md`: có/chưa có; mô tả 8 KiB pages, arena B+Tree, direct I/O.
- `src/main.rs`: REPL dùng Table::create, lex, parse, execute; không gọi Transaction/Wal.
- `src/executor.rs`: INSERT, SELECT WHERE id; SELECT * chưa implement.
- `src/table.rs`: heap + BPlusTree; insert/get; Table::open rebuild index từ heap.
- `src/wal.rs`: Transaction::commit ghi WAL và fsync trước khi apply rows vào Table; rollback bỏ pending buffer.

Không đưa Buffer Pool, query optimizer, MVCC hay full ARIES vào implementation. Không gọi REPL là network server. File source được đọc, không chạy binary hay test trong repo gốc.

## Isolation

Thêm `designs/architecture/`, `public/architecture/`, component riêng và điểm gắn trên index theo yêu cầu. Không commit/push/deploy. `public/` sẽ được copy khi build/deploy toàn portfolio; `noindex` ở preview chỉ là chỉ dẫn crawler, không phải quyền riêng tư.
