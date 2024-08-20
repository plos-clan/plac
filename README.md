# plac

Plant-OS 的音频压缩

**你应该知道：plac 不是任何现代音频压缩格式的替代品，它只是我们尝试创建的一个小型音频格式，你不应该对 plac 的音质抱有过高的期待。**

如果你正在寻找最佳的音频格式，我们推荐 m4a 和 opus。

# 编译命令

linux 环境

```sh
cc plac-player-linux.c -o plac-player -Ofast -lm -lasound
cc plac-encoder-linux.c -o plac-encoder -Ofast -lm -lavcodec -lavformat -lavutil -lswresample
```

windows 环境

未实现，你可以实现后提交 pr

# 资源下载地址

[蓝奏云](https://wwyp.lanzoul.com/b002u8dyyd) (下载后请自行删除文件的 `.txt` 后缀)<br>
密码: dy9j

[NextCloud](https://copi144.eu.org:2000/index.php/s/doZgrjGMsqZDBdN) 推荐

# 反馈或讨论问题

请在 Plant-OS 的 [储存库](https://github.com/plos-clan/Plant-OS) 中提出 Issue

请在 Plant-OS 的 QQ 群 994106260 中进行讨论

也可以发送邮件到 Yan.Huang24@student.xjtlu.edu.cn
