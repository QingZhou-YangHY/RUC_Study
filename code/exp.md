## 实验

代码和脚本写在/home下面，然后模型和数据存在/media/chenzhipeng下面，就不会有奇奇怪怪的问题
/home的大小比较有限，/media的空间大一些


查看环境列表:conda env list
激活环境:conda activate xxx
退出虚拟环境:conda deactivate

查看现存会话：tmux ls
创建会话:tmux new -s <session-name>
切换到指定会话：tmux attach -t <会话名>
分离当前会话：Ctrl+b 然后按 d
杀死当前会话：Ctrl+b 然后按 : 输入 kill-session 或直接运行 tmux kill-session

查看GPU情况在help会话里面
nvidia-smi查看进程/信息,Ctrl + c退出
nvitop更详细似乎,按q退出

查看CUDA:nvcc --version


---

## 文章

主要关注motivation、approach、experiment、limitation这四个点，格式其实不太重要

论文后面有实验的相关设定