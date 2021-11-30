# useful tips
11/30
## check GPU
​		nvidia -smi 
​		nvidia -smi -q
 

## check PID

### ps
    ps -A 查看所有进程
    ps -e ~
    ps -u (name) 查看用户name的进程

### top
    top 实时查看所有进程

### pstree
    pstree 数状图显示
    ps axjf 更加全面的信息
    ps -eLf 获得线程信息
    ps -eo euser,ruser,suser,fuser,f,comm,label 获得安全信息
    top -b -n1 > /tmp/process.log 进程快照储存到文件里
    
### pgrep
    pgrep firefox 查找进程
    pgrep -u (name) 
    ps -elf | grep 2862

### htop atop
    top (plus
    atop 硬件

### ip
    ifconfig  or  ip addr 查看ip地址
    ping + ip 查看延迟
