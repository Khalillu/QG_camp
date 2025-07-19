## Docker

### 概念：

Docker是一个用Go语言实现的开源项目，可以让我们方便的创建和使用容器，Docker将程序以及程序所有的依赖都打包到Docker Container，这样你的程序可以在任何环境都会有一致的表现，这里程序运行的依赖也就是容器就好比集装箱，容器所处的操作系统环境就好比货船或港口，**程序的表现只和集装箱有关系（容器），和集装箱放在那个货船或者哪个港口（操作系统）没有关系。**

**只要将程序打包到Docker中，无论运行在什么环境下程序的行为都是一致的**

### 使用方式

**Dockerfile**

**image**

**container**



Image可理解为可执行程序，Container就是运行起来的进程。

“编写”Image需要Dockerfile，Dockerfile就是Image的源代码，Docker就是“编译器”。

**我们需要做的是：**

在**Dockerfile**中**指定**需要哪些程序、依赖什么样的配置，之后把**Dockerfile交给**“编译器”**Docker**进行“编译”，也就是**Docker build**命令，生成的可执行程序就是**Image**，之后就可以**运行这个Image**，这就是**Docker run**命令，**Image运行起来**后就是**Docker container。**

### 工作方式

Docker使用了常见的CS架构--Client-Server模式，Docker Client负责处理用户输入的各种命令，比如Docker Build、Docker run，真正工作的其实是Server，也就是Docker demon，值得注意的是，Docker client和Docker demon可以运行在同一台机器上。

#### Docker build

写完Dockerfile交给Dockerr“编译”时使用这个命令，那么Client在接收到请求后转发给Docker daemon，接着Docker daemon根据Dockerfile创建出“可执行程序”Image

![img](https://pic1.zhimg.com/v2-f16577a98471b4c4b5b1af1036882caa_r.jpg)

#### Docker run

有了“可执行程序”Image之后就可以运行程序了，**使用命令Docker run**，Docker daemon接收到该命令后找到具体的Image，然后**加载到内存开始执行**，**Image执行起来**就是所谓的**Container**。

![img](https://pic2.zhimg.com/v2-672b29e2d53d2ab044269b026c6bc473_r.jpg)

#### Docker pull

Docker Hub相当于Docker官方的应用商店，上面有别人编写好的Image，如此就不需要自己编写Dockerfile。

**Docker registry用于存放各种Image**，公共的可以供任何人下载Image仓库，就是Docker Hub。那么**Docker pull命令就是从Docker Hub下载Image的方式。**

用户通过Docker client发送命令，Docker daemon接收到命令后向Docker registry发送Image下载请求，下载后存放在本地，这样我们就可以使用Image了。

![img](https://pica.zhimg.com/v2-dac570abcf7e1776cc266a60c4b19e5e_r.jpg)

###  底层实现

#### **基于Linux内核实现**

##### NameSpace机制：

Linux中PID、IPC、网络等资源是全局的，NameSpace是一种资源隔离方案，在该机制下这些资源就不再是全局的了，而是属于某个特定的NameSpace，各个NameSpace下的资源互不干扰，这就使得每个NameSpace看上去像一个独立的操作系统。

##### Control groups机制：

虽有有了NameSpace技术可以实现资源隔离，但进程还是可以不受控的访问系统资源，比如CPU、内存、磁盘、网络等，为了控制容器中进程对资源的访问，Docker采用control groups技术（cgroup），有了cgroup就可以控制容器中进程对系统资源的消耗了，比如限制某个容器使用内存的上限，可以在那些CPU上运行等。