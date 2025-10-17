---
title: "Git và GitHub: Cỗ Máy Thời Gian Cho Code Của Bạn"
description: "Hướng dẫn toàn diện về Git và GitHub: quản lý phiên bản, cộng tác, branching, merging, và workflow thực tế cho developer."
pubDatetime: 2025-06-28T10:00:00Z
tags:
  - git
  - github
  - version-control
  - week2
draft: false
---

# Git và GitHub: Cỗ Máy Thời Gian Cho Code Của Bạn


---

## Giới Thiệu

Bạn đã bao giờ ước mình có một cỗ máy thời gian cho code chưa? Một cách để quay lại các phiên bản trước, hiểu rõ mọi thay đổi, và cộng tác mượt mà mà không làm ảnh hưởng đến công việc của người khác? Đó chính xác là những gì Git và GitHub mang lại!

Hướng dẫn này sẽ đưa bạn qua các khái niệm cốt lõi, quy trình làm việc thực tế và các tính năng nâng cao của Git và GitHub - những công cụ không thể thiếu trong thế giới phát triển phần mềm hiện đại.

---

## Phần 1: Hiểu Về Git - Cốt Lõi Của Hệ Thống Quản Lý Phiên Bản

### Git Là Gì?

Git là một **Hệ thống Quản lý Phiên bản Phân tán (DVCS)**. Điều này có nghĩa là nó giúp bạn theo dõi và lưu trữ các thay đổi trong file theo thời gian, ghi chú ai đã thay đổi gì và khi nào. Khác với các hệ thống CVCS cũ, nơi một server duy nhất chứa tất cả dữ liệu và có nguy cơ mất mát nếu server gặp sự cố, Git cho phép mọi developer có một bản sao đầy đủ lịch sử dự án trên máy local.

### Nguồn Gốc Của Git

Git được tạo ra vào năm 2005 bởi **Linus Torvalds** - người tạo ra Linux, do tranh chấp về việc sử dụng công cụ BitKeeper. Ông thiết kế Git để nhanh, đơn giản, dễ dàng tạo nhánh và có khả năng làm việc phân tán. Trong vòng vài tháng, nó trở thành công cụ chính thức cho việc phát triển Linux kernel và nhanh chóng được cộng đồng mã nguồn mở chấp nhận rộng rãi.

### Nguyên Tắc Hoạt Động Của Git

#### 1. **Snapshot, Không Phải Diffs**
Khác với các VCS cũ chỉ lưu trữ sự khác biệt giữa các phiên bản file, Git lưu trữ một "snapshot" hoàn chỉnh của toàn bộ dự án mỗi khi bạn commit. Nếu một file không thay đổi, Git chỉ đơn giản liên kết đến snapshot trước đó, tiết kiệm không gian trong khi vẫn duy trì tốc độ. Hãy nghĩ Git như một "máy ảnh" chụp ảnh dự án của bạn tại mỗi điểm lưu.

#### 2. **Tính Toàn Vẹn Dữ Liệu Với SHA-1**
Git bảo vệ dữ liệu của bạn bằng hàm băm SHA-1. Mọi đối tượng trong Git đều có một định danh duy nhất 40 ký tự. Nếu nội dung thay đổi, hash sẽ thay đổi hoàn toàn, đảm bảo dữ liệu không bị hỏng.

#### 3. **Nguyên Tắc "Offline-first"**
Git cho phép bạn làm việc mà không cần kết nối internet. Hầu hết các thao tác như commit, tạo nhánh, hay merge đều được thực hiện locally trên máy của bạn. Bạn chỉ cần internet khi muốn chia sẻ code (push/pull).

---

## Phần 2: Quy Trình Git Cơ Bản - Các Lệnh Hàng Ngày

### 1. Cài Đặt Và Cấu Hình

Đầu tiên, bạn cần cài đặt Git cho hệ điều hành của mình. Sau đó, cấu hình thông tin cơ bản:

```bash
# Đặt tên người dùng
git config --global user.name "Tên Của Bạn"

# Đặt email
git config --global user.email "email@example.com"

# Đặt editor mặc định (ví dụ VS Code)
git config --global core.editor "code --wait"
```

#### **Tối Ưu Hóa Hiệu Suất Với Aliases**
Để tiết kiệm thời gian, bạn có thể tạo aliases (phím tắt) cho các lệnh Git thường dùng:

```bash
git config --global alias.st status      # git st thay vì git status
git config --global alias.co checkout    # git co thay vì git checkout
git config --global alias.cm "commit -m" # git cm "Tin nhắn" thay vì git commit -m
git config --global alias.br branch      # git br thay vì git branch
```

Bạn có thể xem cấu hình hiện tại bằng `git config --list` hoặc `git config --list --show-origin`.

### 2. Khởi Tạo Và Clone Repository

#### **Bắt Đầu Dự Án Mới**
Điều hướng đến thư mục dự án và sử dụng `git init`. Điều này tạo ra thư mục ẩn `.git/` để lưu trữ dữ liệu của Git.

```bash
cd /path/to/your/project
git init
```

#### **Làm Việc Với Dự Án Có Sẵn**
Sử dụng `git clone <repository_url>` để tải xuống toàn bộ lịch sử dự án và các nhánh từ server từ xa.

```bash
git clone https://github.com/username/repository.git
```

### 3. Theo Dõi Và Lưu Trữ Thay Đổi

Files trong Git có bốn trạng thái:

- **Untracked**: File mới chưa được Git biết đến
- **Modified**: File hiện có đã được thay đổi nhưng chưa được staged
- **Staged**: File được đánh dấu để đưa vào commit tiếp theo
- **Committed**: File được lưu trữ an toàn trong lịch sử Git

#### **Quản Lý Thay Đổi Code**

```bash
# Kiểm tra trạng thái
git status

# Stage files
git add <tên_file>    # Thêm file cụ thể
git add .             # Thêm tất cả thay đổi

# Commit changes
git commit -m "Mô tả ngắn gọn về thay đổi"
```

#### **Loại Trừ Files Với .gitignore**
File `.gitignore` chỉ định các file và thư mục mà Git nên bỏ qua. Điều này rất quan trọng cho các file tạm thời, thư mục build, file cấu hình cá nhân (như .env), hoặc thư mục thư viện lớn (ví dụ: node_modules).

```bash
# Ví dụ loại trừ file __pycache__
echo __pycache__/ > .gitignore
```

### 4. Xem Lịch Sử

Lệnh `git log` hiển thị lịch sử commit, cho thấy hash commit, tác giả, ngày và tin nhắn.

```bash
git log                    # Lịch sử cơ bản
git log -p                 # Hiển thị diff của mỗi commit
git log --stat             # Hiển thị file và dòng thay đổi trong mỗi commit
git log --oneline --graph --all  # Xem đồ họa tất cả nhánh
```

### 5. Hoàn Tác Thay Đổi

Git cung cấp các lệnh để hoàn tác hoặc sửa đổi thay đổi ở các giai đoạn khác nhau:

```bash
# Hoàn tác thay đổi chưa staged
git restore <file>         # Cẩn thận: Xóa thay đổi chưa commit!

# Hoàn tác thay đổi đã staged
git restore --staged <file>

# Sửa commit cuối
git commit --amend

# Hoàn tác một commit
git revert <commit-hash>

# Lưu tạm thời thay đổi
git stash                  # Lưu thay đổi
git stash pop              # Áp dụng lại thay đổi
```

**⚠️ Cẩn thận với `git reset --hard`**: Lệnh này xóa vĩnh viễn thay đổi chưa commit và quay về trạng thái cũ.

---

## Phần 3: Làm Việc Với Remote Repository - Yếu Tố Cộng Tác

Remote repository là phiên bản dự án được lưu trữ trên server (như GitHub, GitLab, hoặc Bitbucket) cho phép cộng tác và đồng bộ hóa.

### Các Lệnh Remote Cơ Bản

```bash
# Xem remote
git remote -v

# Thêm remote
git remote add origin <URL>

# Push thay đổi
git push origin main

# Pull thay đổi
git pull origin main      # Kết hợp fetch và merge
git fetch origin          # Chỉ tải thông tin, không merge
```

### Tagging - Đánh Dấu Phiên Bản Quan Trọng

Tags đánh dấu các điểm quan trọng trong lịch sử repository, thường cho các phiên bản release.

```bash
# Annotated tags (khuyến nghị cho release)
git tag -a v1.0 -m "Version 1.0 - Official Release"

# Lightweight tags
git tag v1.0-beta

# Push tags
git push origin v1.0              # Push tag cụ thể
git push origin --tags            # Push tất cả tags
```

---

## Phần 4: Quản Lý Nhánh - Phát Triển Song Song Và Lịch Sử Sạch

Nhánh là nền tảng của Git, cho phép phát triển song song. Một nhánh về cơ bản là một con trỏ đến một commit cụ thể.

### Quy Trình Phát Triển Tính Năng

```bash
# Tạo và chuyển nhánh
git checkout -b feature/login

# Phát triển và commit
# ... làm việc trên tính năng ...
git add .
git commit -m "Thêm tính năng đăng nhập"

# Merge vào main
git checkout main
git merge feature/login
```

### Giải Quyết Xung Đột

Đôi khi Git không thể tự động merge thay đổi (ví dụ: cùng một dòng được thay đổi khác nhau trên cả hai nhánh). Git sẽ đánh dấu những xung đột này, và bạn cần chỉnh sửa file thủ công để giải quyết. Sau khi giải quyết, `git add .` và `git commit` để hoàn thành merge.

### Quản Lý Nhánh

```bash
# Liệt kê nhánh
git branch              # Nhánh local, * đánh dấu nhánh hiện tại
git branch -r           # Remote-tracking branches
git branch -a           # Tất cả nhánh (local và remote)

# Xóa nhánh
git branch -d <tên_nhánh>    # Xóa nhánh đã merge
git branch -D <tên_nhánh>    # Force xóa nhánh chưa merge
git push origin --delete <tên_nhánh>  # Xóa nhánh trên remote

# Đổi tên nhánh
git branch -m <tên_cũ> <tên_mới>
git branch -m <tên_mới>      # Đổi tên nhánh hiện tại
```

### Rebase - Tạo Lịch Sử Sạch

Rebase là tính năng mạnh mẽ của Git tạo ra lịch sử commit sạch, tuyến tính bằng cách "tái áp dụng" commits từ nhánh hiện tại lên đầu nhánh khác.

```bash
# Rebase cơ bản
git checkout feature-branch
git rebase main

# Interactive Rebase
git rebase -i HEAD~3    # Chỉnh sửa 3 commit cuối
```

**⚠️ Cảnh báo quan trọng**: Không bao giờ rebase các nhánh đã được chia sẻ công khai. Rebase thay đổi lịch sử commit, có thể gây ra vấn đề nghiêm trọng cho người cộng tác.

---

## Phần 5: Quy Trình Nhánh Phổ Biến - Cấu Trúc Cộng Tác

### 1. GitFlow
Được giới thiệu bởi Vincent Driessen, GitFlow định nghĩa các loại nhánh cụ thể và quy tắc tương tác.

- **Main branch**: Code sẵn sàng production, được tag với số phiên bản
- **Develop branch**: Nhánh tích hợp cho tất cả tính năng hoàn thành
- **Feature branches**: Tạo từ develop cho tính năng mới
- **Release branches**: Tạo từ develop khi chuẩn bị release
- **Hotfix branches**: Tạo từ main cho sửa lỗi khẩn cấp

### 2. Trunk-based Development
Phương pháp đơn giản hơn nơi công việc diễn ra trực tiếp trên nhánh "trunk" (main). Sử dụng feature branches ngắn hạn, dựa vào feature flags để kiểm soát tính năng mới.

### 3. GitHub Flow
Quy trình tối ưu cho continuous deployment:

1. Tạo nhánh từ main cho mỗi tính năng/bug
2. Commit và push thường xuyên
3. Mở Pull Request (PR) để review
4. Merge sau khi pass checks và được approve
5. Deploy ngay sau khi merge

### 4. GitLab Flow
Kết hợp GitHub Flow với quản lý môi trường. Sử dụng production branches thay vì chỉ tags và thêm environment branches (staging, pre-production).

---

## Phần 6: Cộng Tác Với GitHub - Nền Tảng Đám Mây

GitHub là nền tảng web lưu trữ Git repositories, cung cấp công cụ cho cộng tác, review code và quản lý dự án.

### 1. Thiết Lập Tài Khoản & Bảo Mật

```bash
# Tạo SSH key
ssh-keygen -t ed25519 -C "email@example.com"

# Thêm key vào GitHub (qua web interface)
# Bật Two-Factor Authentication (2FA)
```

### 2. Quy Trình Fork & Pull Request

Đây là cách tiêu chuẩn để đóng góp cho các dự án bạn không có quyền write trực tiếp.

#### **Các Bước Đóng Góp Cho Dự Án Mã Nguồn Mở**

1. **Fork**: Tạo bản sao cá nhân của repository trên tài khoản GitHub
2. **Clone**: Clone repository đã fork về máy local
   ```bash
   git clone https://github.com/your-username/repository.git
   ```
3. **Set Upstream**: Thêm repository gốc làm "upstream" remote
   ```bash
   git remote add upstream https://github.com/original-owner/repository.git
   ```
4. **Tạo Nhánh**: Tạo nhánh topic mới từ main cho thay đổi của bạn
   ```bash
   git checkout -b fix-login-bug
   ```
5. **Thay Đổi & Push**: Thực hiện thay đổi, commit và push nhánh mới
   ```bash
   git push origin fix-login-bug
   ```
6. **Tạo Pull Request**: Đi đến repository gốc trên GitHub, click "New pull request"
7. **Thảo Luận & Cập Nhật**: Chủ sở hữu dự án sẽ review PR và có thể yêu cầu thay đổi

### 3. Quản Lý Repository Trên GitHub

- **Cấu hình**: Tạo repository mới, đặt tên, mô tả và quyền riêng tư
- **Tài liệu**: Cung cấp README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md
- **Templates**: Tạo templates trong thư mục .github
- **Branch Protection Rules**: Cấu hình rules trong Settings > Branches
- **GitHub Pages**: Xuất bản tài liệu dự án trực tiếp từ repository

### 4. Tổ Chức & Teams

Organizations quản lý nhiều repositories và users. Teams trong organization nhóm thành viên và gán quyền truy cập (Read, Write, Admin) cho repositories.

### 5. Tự Động Hóa & Tích Hợp

#### **GitHub Actions**
Hệ thống CI/CD tích hợp của GitHub để tự động hóa testing, building và deploying.

```yaml
# Ví dụ workflow đơn giản
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run tests
      run: npm test
```

#### **Webhooks & API**
- **Webhooks**: Cho phép GitHub gửi HTTP POST notifications đến external services
- **GitHub REST API**: Cho phép tương tác programmatic với GitHub

### Mẹo Cộng Tác Hiệu Quả Trên GitHub

- Sử dụng **Draft Pull Requests** để nhận feedback sớm
- Tạo **Review Checklists** để đảm bảo chất lượng code nhất quán
- **Sync Fork thường xuyên** để tránh conflicts
  ```bash
  git fetch upstream
  git checkout main
  git merge upstream/main
  git push origin main
  ```

---

## Phần 7: Tùy Chỉnh Nâng Cao - Điều Chỉnh Git Cho Nhu Cầu Của Bạn

### 1. Cấu Hình .gitconfig Nâng Cao

```bash
# Đặt tên nhánh mặc định
git config --global init.defaultBranch main

# Template commit message
git config --global commit.template ~/.gitmessage.txt

# Credential helper
git config --global credential.helper store    # Lưu vĩnh viễn
git config --global credential.helper cache    # Lưu tạm thời
```

### 2. Git Attributes (.gitattributes)

File `.gitattributes` định nghĩa cách Git xử lý các loại file khác nhau.

```bash
# Line endings
*.txt text                    # Normalize to LF in repo
*.bat text eol=crlf          # Convert to CRLF on checkout

# Binary files
*.png binary                  # Mark as binary

# Merge strategies
*.config merge=ours          # Always keep local version
```

### 3. Git Hooks - Tự Động Hóa Quy Trình

Git Hooks là scripts chạy tự động tại các điểm cụ thể trong quy trình Git.

#### **Client-side Hooks**
- **pre-commit**: Chạy trước commit, hữu ích cho linting code hoặc chạy tests
- **commit-msg**: Kiểm tra format của commit message
- **pre-push**: Chạy trước push, hữu ích cho final checks

#### **Server-side Hooks**
- **pre-receive**: Chạy khi server nhận push request
- **post-receive**: Chạy sau khi push hoàn thành, thường dùng để trigger CI/CD

### 4. Aliases & Scripts Nâng Cao

```bash
# Pretty log alias
git config --global alias.lg "log --graph --pretty=format:'%C(auto)%h%d %s %C(black)%C(bold)%cr %C(black)[%an]' --abbrev-commit"

# Create and switch branch alias
git config --global alias.nb "!f() { git checkout -b $1; }; f"

# Cleanup merged branches alias
git config --global alias.cleanup "!git branch --merged | grep -v '\\*' | xargs -n 1 git branch -d"
```

---

## Kết Luận

Thành thạo Git và GitHub là kỹ năng thiết yếu cho bất kỳ ai làm việc với code, đặc biệt trong các dự án Data Science và Machine Learning như được nhấn mạnh trong lộ trình AIO2025. Từ dự báo bán hàng đến phân khúc khách hàng, quản lý phiên bản hiệu quả đảm bảo dự án của bạn được quản lý tốt, có tính cộng tác cao, và mọi thay đổi đều được theo dõi và có thể khôi phục.

Git và GitHub không chỉ là công cụ - chúng là nền tảng của phát triển phần mềm hiện đại, cho phép bạn làm việc với sự tự tin rằng mọi thay đổi đều được bảo vệ và có thể truy ngược lại. Hãy bắt đầu hành trình làm chủ "cỗ máy thời gian" cho code của bạn ngay hôm nay!

---

*Tài liệu này được biên soạn dựa trên nội dung từ độii ngữ giảng viên trong chương trình AIO2025, kết hợp với kinh nghiệm thực tế trong phát triển phần mềm và quản lý dự án.* 