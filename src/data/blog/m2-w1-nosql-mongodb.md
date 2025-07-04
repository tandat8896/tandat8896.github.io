---
title: "Database NoSQL (1): Khởi đầu với MongoDB cho AI/ML"
description: "Tìm hiểu về cơ sở dữ liệu NoSQL, lý do chọn MongoDB, thao tác cơ bản và ứng dụng thực tiễn trong AI/ML. Bài viết dành cho người mới bắt đầu."
pubDatetime: 2025-07-03T01:27:00Z
tags:
  - mongodb
  - nosql
  - database
  - ai
  - week1
  - module2
draft: false
---

Chào bạn,
Hành trình khám phá thế giới cơ sở dữ liệu NoSQL, đặc biệt là MongoDB, là một trải nghiệm thú vị, mở ra một cách tiếp cận linh hoạt hơn trong việc quản lý và lưu trữ dữ liệu. Dưới đây là hành trình học tập về MongoDB NoSQL, được tổng hợp từ các kiến thức đã học.

--------------------------------------------------------------------------------
Bắt đầu với NoSQL: Khi dữ liệu không còn là những bảng cố định
Trong thời đại số, dữ liệu ngày càng đa dạng và phức tạp, không còn chỉ giới hạn trong các khuôn mẫu cố định. Đây chính là lúc chúng ta cần tìm hiểu về NoSQL.
Vậy, cơ sở dữ liệu là gì? Cơ sở dữ liệu là một bộ sưu tập thông tin có tổ chức, có thể là số, văn bản hoặc các loại phương tiện khác. Nó cho phép chúng ta dễ dàng truy cập, khôi phục, quản lý và cập nhật dữ liệu. Để quản lý cơ sở dữ liệu hiệu quả, chúng ta cần một Hệ quản trị cơ sở dữ liệu (DBMS) – một loại phần mềm giúp quản lý cơ sở dữ liệu.

SQL vs NoSQL: Sự khác biệt cốt lõi
Các hệ cơ sở dữ liệu truyền thống như SQL (Structured Query Language) lưu trữ dữ liệu bằng mô hình dữ liệu quan hệ, tức là dữ liệu được tổ chức dưới dạng các bảng với hàng và cột cố định. Điều này rất phù hợp khi cấu trúc dữ liệu rõ ràng và ít thay đổi.
Tuy nhiên, thực tế phát triển phần mềm cho thấy, nhu cầu thay đổi cấu trúc dữ liệu thường xuyên là điều khó tránh khỏi. Ví dụ, một sinh viên ban đầu chỉ cần 3 thông tin, sau đó lại cần bổ sung thêm 4 thông tin khác. Nếu dùng SQL, việc thêm cột, xóa cột sẽ rất khó xử lý và đòi hỏi mô hình phải linh hoạt hơn. Chính vì lý do này, NoSQL ra đời.
NoSQL là viết tắt của "No Relational", "No RDBMS", hoặc "Not Only SQL". Điểm khác biệt lớn nhất là NoSQL lưu trữ dữ liệu một cách khác biệt so với SQL, không theo định dạng bảng truyền thống. Dữ liệu thường được tổ chức dưới dạng {"key": "value"}. Sự linh hoạt này là lý do chính khiến NoSQL trở thành một lựa chọn hấp dẫn khi mô hình dữ liệu cần sự thay đổi liên tục.

Các loại hình cơ sở dữ liệu NoSQL phổ biến
Có nhiều loại hình NoSQL, mỗi loại phù hợp với một số trường hợp sử dụng nhất định:
• Key-Value Store Database: Đây là loại cơ sở dữ liệu đơn giản nhất, giống như một bảng băm (hash table) hoặc từ điển (dictionary) trong lập trình. Nó được sử dụng chủ yếu khi tất cả các truy cập vào cơ sở dữ liệu đều thông qua một khóa chính duy nhất. Ví dụ: lưu Name: "Thai" với giá trị "Thai", hoặc Age: 20 với giá trị 20.
• Graph Database: Cơ sở dữ liệu đồ thị tập trung vào việc biểu diễn mối quan hệ giữa các thực thể. Mọi thứ được lưu trữ dưới dạng cạnh (edge), nút (node) hoặc thuộc tính (attribute). Mỗi nút và cạnh có thể có nhiều thuộc tính. Ví dụ: biểu diễn mối quan hệ Thai -Is_Friend_With-> Anh hoặc AI -Like-> Nghe An.
• Document Database: Đây là loại NoSQL mà MongoDB thuộc về. Nó lưu trữ dữ liệu dưới dạng các tập hợp khóa/giá trị có cấu trúc lỏng lẻo trong các tài liệu (documents), thường sử dụng các định dạng chuẩn như XML, JSON (JavaScript Object Notation) hoặc BSON. Các tài liệu được truy cập trong cơ sở dữ liệu thông qua một khóa duy nhất và được xử lý như một thể thống nhất, không bị chia nhỏ.

MongoDB: Trái tim của cơ sở dữ liệu tài liệu
MongoDB là một trong những hệ cơ sở dữ liệu tài liệu (document database) phổ biến nhất.

Cấu trúc dữ liệu trong MongoDB
Trong MongoDB, dữ liệu được lưu trữ dưới dạng các cặp trường-giá trị (field-value) bên trong các tài liệu (documents). Các tài liệu này được lưu trữ trong collection (tập hợp các tài liệu), và nhiều collection được lưu trữ trong một database.
• Tài liệu (Document): Một tài liệu trong MongoDB sử dụng định dạng JSON, ví dụ:
```json
{
  "_id": 1,
  "name": "Thai",
  "age": 20,
  "address": { "city": "Hanoi", "zip": "10000" },
  "skills": ["math", "english"]
}
```
• Trường _id là một định danh duy nhất cho mỗi tài liệu trong một collection.
• Kiểu dữ liệu linh hoạt: MongoDB hỗ trợ nhiều kiểu dữ liệu khác nhau, bao gồm:
  ◦ Numerical: Các số nguyên và số thập phân.
  ◦ Date: Giá trị ngày và thời gian.
  ◦ String: Giá trị ký tự/văn bản.
  ◦ Ngoài ra, cấu trúc tài liệu trong MongoDB rất linh hoạt (polymorphic). Nó cho phép các tài liệu trong cùng một collection không cần có cấu trúc giống hệt nhau. Bạn có thể dễ dàng thêm các cặp trường-giá trị mới vào một tài liệu riêng lẻ.
  ◦ MongoDB cũng hỗ trợ Nested Document (tài liệu lồng nhau), nơi một trường có thể chứa một tài liệu khác, ví dụ:
```json
{
  "name": "Anh",
  "address": { "city": "HCM", "zip": "70000" }
}
```
  ◦ Ngoài ra, các tài liệu có thể chứa mảng (array), một cấu trúc dữ liệu chứa một tập hợp các phần tử, ví dụ:
```json
{
  "name": "Hoa",
  "skills": ["math", "english", "python"]
}
```

MongoDB được ứng dụng trong nhiều hệ thống AI thực tế, chẳng hạn như lưu thông tin ảnh trong hệ thống nhận diện khuôn mặt, lưu trữ vector của ảnh hoặc văn bản để làm AI gợi ý sản phẩm, hoặc quản lý dữ liệu phức tạp cho các giao dịch gian lận. Bất kỳ dữ liệu phi cấu trúc nào cũng có thể được xem xét để lưu trữ trong MongoDB.

Bắt đầu thực hành: Cài đặt và kết nối MongoDB
Để làm việc với MongoDB, chúng ta có một hệ sinh thái mạnh mẽ bao gồm các phiên bản Community Edition, Enterprise Edition, và các công cụ như MongoDB Atlas, MongoDB Shell, MongoDB Drivers, BI Connectors, và Compass.

Cài đặt và cấu hình MongoDB Atlas (Phiên bản đám mây)
MongoDB Atlas là nền tảng lưu trữ và quản lý cơ sở dữ liệu MongoDB trên đám mây.
1. Tạo tài khoản MongoDB Atlas: Truy cập trang chủ MongoDB Atlas và đăng ký tài khoản miễn phí.
2. Tạo Cluster miễn phí: Sau khi đăng nhập, chọn "Build a Database", chọn gói "Shared - Free" (miễn phí) và khu vực gần bạn (ví dụ: AWS Singapore). Dung lượng miễn phí thường là 512MB. Đặt tên cho Cluster của bạn.
3. Thêm người dùng Database (Database User): Vào mục "Database Access", thêm người dùng mới với tên và mật khẩu. Bạn có thể cấp quyền "Read and write to any database".
4. Cấu hình IP Access: Vào mục "Network Access". Để cho phép truy cập từ mọi địa chỉ IP (chỉ nên dùng cho môi trường học tập hoặc demo), nhập 0.0.0.0/0. Hoặc bạn có thể nhập chính xác địa chỉ IP hiện tại của máy bạn.

Kết nối VSCode với MongoDB
VSCode là một trình soạn thảo mạnh mẽ, hỗ trợ tốt việc kết nối và làm việc với MongoDB.
1. Cài đặt Extension: Mở VSCode, vào mục Extensions (Ctrl + Shift + X), tìm và cài đặt "MongoDB for VSCode".
2. Lấy Connection String: Truy cập Cluster MongoDB Atlas của bạn, vào mục "Connect", chọn "MongoDB for VS Code" để lấy chuỗi kết nối (connection string). Chuỗi này có dạng như: mongodb+srv://<username>:<password>@<cluster-url>/test.
3. Tạo kết nối trong VSCode: Mở mục MongoDB ở thanh sidebar của VSCode, chọn "Connect", nhập connection string và nhấn "Connect".
4. Kiểm tra kết nối và thao tác cơ bản: Sau khi kết nối thành công, bạn sẽ thấy danh sách database và collection trong sidebar. Bạn có thể chạy các lệnh cơ bản như:
5. Lưu ý kiểm tra lại connection string, username, password và IP Access nếu gặp lỗi kết nối.

Sử dụng MongoDB Shell và Database Tools
Bạn cũng có thể tương tác với MongoDB bằng cách cài đặt MongoDB Shell hoặc Database Tools. Ví dụ, để nhập (import) hoặc xuất (export) dữ liệu, bạn có thể dùng mongoimport và mongoexport.
• mongoexport: mongoexport --uri="<connection_string>" --collection=<collection_name> --out=<output_file>
• mongoimport: mongoimport --uri="<connection_string>" --collection=<collection_name> --file=<input_file>

Ví dụ lệnh import file JSON:
```bash
./mongodb-database-tools-windows-x86_64-100.12.2/bin/mongoimport --uri="<connection_string>" --collection=companies --file=sample/companies.json --jsonArray
```

Ngôn ngữ truy vấn Mongo (MQL): Làm chủ dữ liệu của bạn
MQL cung cấp một bộ các lệnh mạnh mẽ để thao tác với dữ liệu trong MongoDB.

1. Tạo / Xóa Database và Collection
• Hiển thị tất cả databases:
```js
show dbs
```
• Tạo/Chuyển sang một database:
```js
use database_name
```
• Xóa database hiện tại:
```js
db.dropDatabase()
```
• Hiển thị collections trong database:
```js
show collections
```
• Tạo collection:
```js
db.createCollection('collection_name')
```
• Xóa collection:
```js
db.collection_name.drop()
```

2. Chèn tài liệu (Inserting Documents)
Sử dụng phương thức insert() hoặc insertOne(), insertMany():
• Chèn một tài liệu:
```js
db.student.insert({"name":"Thai"})
```
• Chèn nhiều tài liệu:
```js
db.student.insert([{ "name": "Thai" }, { "age": 20 }])
```
Lưu ý: Trường _id phải là duy nhất trong mỗi collection. Nếu bạn cố gắng chèn hai tài liệu với cùng một _id, MongoDB sẽ báo lỗi.

Ví dụ thực tế:
```js
db.companies.insertOne({ "name": "AI Startup", "number_of_employees": 10 })
db.grades.insertMany([
  { "student_id": 1, "scores": [ { "type": "exam", "score": 95 } ] },
  { "student_id": 2, "scores": [ { "type": "quiz", "score": 88 } ] }
])
```

3. Tìm kiếm tài liệu (Finding Documents)
• findOne(query, projection): Trả về một tài liệu đầu tiên khớp với điều kiện.
◦ db.student.findOne(): Trả về tài liệu đầu tiên trong collection.
◦ db.student.findOne({"name":"Anh"}): Tìm một tài liệu có name là "Anh".
• find(query, projection): Trả về tất cả các tài liệu khớp với điều kiện.
◦ db.student.find(): Trả về tất cả tài liệu.
◦ db.student.find({"name":"Anh", "age":20}): Tìm tất cả tài liệu có name là "Anh" VÀ age là 20.
◦ Truy vấn tài liệu lồng nhau: Sử dụng ký hiệu dấu chấm (.) để truy vấn các trường trong tài liệu lồng nhau. Ví dụ, để tìm tất cả tài liệu trong collection "Inspection" với zip code là 11385 trong trường address:
```js
db.inspections.find({ "address.zip": 11385 })
```

4. Toán tử truy vấn quan trọng
• Toán tử so sánh: Sử dụng cú pháp {field: {operator: value}}.
◦ $eq: Bằng (equal to). Ví dụ: {"salary":{$eq:50000}}
◦ $lt: Nhỏ hơn (less than). Ví dụ: {"salary":{$lt:50000}}
◦ $gt: Lớn hơn (greater than).
◦ $ne: Không bằng (not equal to).
◦ $lte: Nhỏ hơn hoặc bằng.
◦ $gte: Lớn hơn hoặc bằng.
◦ $in: Trong tập hợp các giá trị. Ví dụ: {"salary":{$in:[40000,50000]}}
◦ $nin: Không trong tập hợp các giá trị.
• Toán tử logic:
◦ $and: Trả về tài liệu khớp với TẤT CẢ các điều kiện. Ví dụ:
```js
db.trips.find({ $and: [ { "tripduration": { $gt: 400 } }, { "birth year": { $gt: 1988 } } ] })
```

**Khuyên dùng explicit thay vì implicit khi viết truy vấn với toán tử $**

Khi viết truy vấn với các toán tử như `$and`, `$or`, `$nor`, bạn nên ưu tiên viết explicit (tường minh) thay vì implicit (liệt kê nhiều điều kiện trong object). Việc này giúp code rõ ràng, dễ đọc, dễ bảo trì và tránh nhầm lẫn khi mở rộng điều kiện.

Ví dụ:
- **Implicit (ngầm định):**
```js
db.trips.find({ "tripduration": { $gt: 400 }, "birth year": { $gt: 1988 } })
```
- **Explicit (tường minh, nên dùng):**
```js
db.trips.find({ $and: [ { "tripduration": { $gt: 400 } }, { "birth year": { $gt: 1988 } } ] })
```

Với truy vấn phức tạp, explicit sẽ giúp bạn dễ dàng thêm, bớt, hoặc kết hợp các điều kiện mà không bị lỗi logic hoặc khó đọc code.

**Tóm lại:** Hãy ưu tiên viết query explicit với các toán tử $ để code rõ ràng, dễ bảo trì!

◦ $or: Trả về tài liệu khớp với BẤT KỲ điều kiện nào. Ví dụ, tìm tài liệu mà trường "start station name" hoặc "end station name" có giá trị null:
```js
db.trips.find({ $or: [ { "start station name": null }, { "end station name": null } ] })
```
◦ $not: Trả về tài liệu không khớp với biểu thức. Ví dụ: db.trips.find({"usertype": {$not:{$eq:"Subscriber"}}}) tương đương với db.trips.find({"usertype": {$ne:"Subscriber"}}).
◦ $nor: Trả về tài liệu không khớp với BẤT KỲ điều kiện nào.
• $expr: Cho phép bạn sử dụng các biểu thức tổng hợp để so sánh các trường với nhau hoặc thực hiện các phép tính phức tạp. Ví dụ, tìm tài liệu mà giá trị của "field1" giống với "field2":
```js
db.collection.find({ $expr: { $eq: [ "$field1", "$field2" ] } })
```
• Toán tử phần tử:
◦ $exists: Trả về tài liệu chứa trường được chỉ định. Ví dụ: db.companies.find({"ipo":{$exists:true}})
◦ $type: Trả về tài liệu mà trường chứa giá trị của một kiểu dữ liệu BSON cụ thể (ví dụ: 2 cho String, 4 cho Array, 10 cho Null). Ví dụ: db.companies.find({"homepage_url":{$type:2}}) để tìm các URL có kiểu String.

5. Phương thức con trỏ (Cursor Methods)
Khi bạn chạy lệnh find(), MongoDB sẽ trả về một con trỏ (cursor) tới tập kết quả. Bạn có thể áp dụng các phương thức sau lên con trỏ:
• count(): Trả về số lượng tài liệu trong tập kết quả (lưu ý: phương thức này đã cũ và có thể không chính xác với con trỏ, nên ưu tiên dùng countDocuments()).
• countDocuments(): Đếm chính xác số lượng tài liệu trong collection mà không cần tải toàn bộ dữ liệu. Nên dùng để đếm tài liệu.
• limit(n): Giới hạn số lượng tài liệu trả về.
• skip(X): Bỏ qua X tài liệu đầu tiên.
• sort({parameters}): Sắp xếp tài liệu dựa trên các trường được chỉ định (1 cho tăng dần, -1 cho giảm dần). Ví dụ: db.trips.find().sort({"tripduration":1}).
• size(): Giống như count(), nhưng cần tải toàn bộ con trỏ nên có thể chậm và tốn RAM với dữ liệu lớn.

Ví dụ thực tế:
```js
db.companies.find({}, { "name": 1, "number_of_employees": 1, "_id": 0 })
  .sort({ "number_of_employees": -1 })
  .limit(10)
```

6. Projection
Sử dụng trong phương thức find(query, projection) để chỉ định các trường muốn trả về trong tài liệu kết quả.
• {"field":1}: Bao gồm trường đó.
• {"field":0}: Loại trừ trường đó.
• Trường _id được bao gồm theo mặc định trừ khi bạn chỉ định "_id":0. Ví dụ, để trả về tên và số lượng nhân viên của 10 công ty hàng đầu theo số lượng nhân viên giảm dần, chỉ hiển thị tên và số lượng nhân viên, không hiển thị _id:
```js
db.companies.find({}, {"name":1,"number_of_employees":1,"_id":0})
    .sort({"number_of_employees":-1})
    .limit(10)
```

7. Truy vấn mảng (Querying Arrays)
• $all: Trả về tài liệu mà một trường kiểu mảng chứa TẤT CẢ các phần tử được chỉ định (thứ tự không quan trọng). Ví dụ:
```js
db.students.find({ skills: { $all: ["math", "english"] } })
```
• $size: Trả về tài liệu mà một trường kiểu mảng có kích thước (số phần tử) khớp với giá trị chỉ định. Ví dụ:
```js
db.students.find({ hobbies: { $size: 2 } })
```
• $elemMatch: Trả về tài liệu chứa một trường mảng với một phần tử khớp với tiêu chí truy vấn được chỉ định. Ví dụ:
```js
db.grades.find({ scores: { $elemMatch: { type: "exam", score: { $gt: 90 } } } })
```

8. Xóa tài liệu (Deleting Documents)
• deleteOne(filter): Xóa một tài liệu đầu tiên khớp với điều kiện.
◦ db.student.deleteOne({"name":"Anh"})
• deleteMany(filter): Xóa tất cả tài liệu khớp với điều kiện.
◦ db.student.deleteMany({"name":"Anh"})

Ví dụ thực tế:
```js
db.companies.deleteMany({ "number_of_employees": { $lt: 5 } })
```

9. Cập nhật tài liệu (Updating Documents)
• updateOne({filter},{update}): Cập nhật một tài liệu duy nhất khớp với bộ lọc.
• updateMany({filter},{update}): Cập nhật tất cả tài liệu khớp với bộ lọc.
◦ Ví dụ: db.student.updateOne({"name":"Thai"}, {$set:{"name":"Anh"}}) để đổi tên "Thai" thành "Anh" cho một tài liệu.
◦ Ví dụ: db.student.updateMany({}, {$set:{"address":"Ha Noi"}}) để thêm trường "address" cho tất cả tài liệu.

Ví dụ thực tế:
```js
db.companies.updateMany({}, { $set: { "country": "Vietnam" } })
db.grades.updateOne({ "student_id": 1 }, { $push: { "scores": { "type": "bonus", "score": 10 } } })
```

10. Toán tử cập nhật (Update Operators)
• $set: Thay thế giá trị của một trường bằng giá trị được chỉ định.
◦ db.student.updateMany({"name":"Hoa"}, {$set:{"address":"Ho Chi Minh"}})
• $unset: Xóa một trường cụ thể.
◦ db.collection.updateMany({filter},{$unset:{field:value,…}})
• $rename: Đổi tên một trường.
◦ db.collection.updateMany({filter},{$rename:{field:value,…}})
• $inc: Tăng giá trị của một trường lên một giá trị cụ thể.
◦ db.student.updateMany({"age":20}, {$inc:{"age":1}})
• $push: Thêm một giá trị vào cuối một mảng.
◦ db.student.updateMany({"name":"Hoa"}, {$push:{"address":"Ha Noi"}})

Ví dụ thực tế:
```js
db.companies.updateMany({ "name": "AI Startup" }, { $inc: { "number_of_employees": 2 } })
db.students.updateMany({ "name": "Hoa" }, { $push: { "skills": "python" } })
```

--------------------------------------------------------------------------------
Với sự linh hoạt và khả năng xử lý dữ liệu phi cấu trúc mạnh mẽ, MongoDB là một công cụ cực kỳ giá trị trong bộ công cụ của bất kỳ nhà phát triển nào, đặc biệt trong các dự án đòi hỏi sự nhanh nhẹn và khả năng mở rộng liên tục.

Hãy thực hành với các file mẫu trong thư mục sample/ để cảm nhận rõ hơn sức mạnh của MongoDB và NoSQL nhé! 