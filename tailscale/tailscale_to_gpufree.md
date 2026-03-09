# 跨越云服务器端口限制：使用 Tailscale 部署 Isaac Sim WebRTC 推流(无法起效，疑似容器不支持tun模式导致的？)

##  背景
在云服务器上运行 NVIDIA Isaac Sim 时，官方的 WebRTC 方案默认需要开放 TCP 49100 和 UDP 47998 端口。
然而，许多云服务商存在严格的安全组限制（例如仅开放 1000-19999 的 TCP 端口，屏蔽 UDP），导致本地 Isaac Sim WebRTC Streaming Client 无法连接或持续黑屏。

**解决方案**：通过部署 [Tailscale](https://tailscale.com/) 构建 Mesh 虚拟局域网（VLAN），将本地电脑与云服务器置于同一内网环境下，利用其底层 WireGuard 协议实现 P2P 直接穿透，从而完美绕过公网端口限制。

---

##  环境适配说明
特别针对 **Docker 容器环境**。由于这类环境通常为了轻量化阉割了 `systemd`（系统主守护进程 PID 1），并且禁用了虚拟网卡设备（TUN/TAP）权限，常规的 `systemctl start tailscaled` 命令会报错。我们需要使用**用户态网络 (Userspace Networking)** 模式来手动启动。

---

## 🛠️ 部署步骤

### 1. 清理潜在的残留进程
如果在之前的尝试中有卡死的后台进程或占用的 socket 文件，会导致新进程启动失败。首先执行清理：
```bash
# 强制结束所有 tailscaled 进程
sudo killall tailscaled

# 删除可能残留的 socket 占用文件
sudo rm -f /var/run/tailscale/tailscaled.sock
```

### 2. 在后台以用户态启动 Tailscale 守护进程
在没有 systemd 和 TUN 权限的容器中，通过以下命令启动 tailscaled。
参数说明：使用用户态网络，开启本地 socks5 代理，并将运行日志重定向到后台以保持终端整洁。

```Bash
sudo tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /tmp/tailscaled.log 2>&1 &
(注：运行后按下 Enter 键即可回到终端输入状态)
```
### 3. 获取授权并登录
守护进程运行后，执行启动命令：

```Bash
sudo tailscale up
```
此时终端会输出一个类似 https://login.tailscale.com/a/xxxxxx 的链接。

复制该链接到你的本地浏览器中打开。

登录你的 Tailscale 账号完成设备授权。

授权成功后，终端会提示 Success。你可以通过 tailscale ip 命令查看分配给该容器的局域网 IP（通常以 100.x.x.x 开头）。

### 4. 启动 Isaac Sim
获取到 Tailscale IP 后，不需要再绑定云服务器的公网 IP。使用分配的虚拟内网 IP 启动 Isaac Sim 的 Headless 模式：

```Bash
./isaac-sim.streaming.sh --/app/livestream/publicEndpointAddress=<你的Tailscale虚拟IP> --/app/livestream/port=49100
```
### 5. 本地客户端连接
确保你的本地电脑（Windows/Mac）也安装并登录了同一个 Tailscale 账号。

打开本地的 Isaac Sim WebRTC Streaming Client。

在 Server 栏中输入 <你的Tailscale虚拟IP>，点击 Connect 即可享受低延迟的实时仿真画面。

## 退出tailscale
### 1. 仅仅是暂时断开连接（推荐）
如果只是想暂时掐断这台服务器与虚拟局域网的连接，但保留登录状态，以便下次随时重连：

```Bash
sudo tailscale down
```
这相当于在手机上把 VPN 的开关拨到“关闭”。下次想用的时候，直接运行 sudo tailscale up 就能秒连，不需要重新扫码或登录。

### 2. 彻底关闭后台进程
如果跑完仿真了，想把刚才后台运行的 tailscaled 守护进程彻底杀掉，释放系统资源：

```Bash
sudo killall tailscaled
```
因为我们是用 & 强制挂在后台的，所以用 killall 是最直接有效的清理方式。

### 3. 彻底注销并解绑设备
如果你以后都不打算在这个云服务器（或者这个特定的 Docker 容器）上用 Tailscale 了，想把它从你的 Tailscale 账号设备列表里删掉：

```Bash
sudo tailscale logout
```
执行这个命令后，下次再启动就需要重新打开浏览器链接进行授权了。

### 最常用的组合：
通常跑完项目准备关机或退出容器时，直接执行一次 sudo killall tailscaled 就可以了，简单粗暴且干净。


##  常见问题排查 (FAQ)
### Q: 运行 sudo tailscale up 提示 failed to connect to local tailscaled？
A: 这说明守护进程没有成功在后台运行。请重新执行本文的【步骤 1】和【步骤 2】。

### Q: 运行守护进程时提示 TPM: error opening...？
A: 这是因为 Docker 容器中没有物理 TPM 加密模块，属于正常现象，不影响 Tailscale 的核心网络功能，直接忽略即可。