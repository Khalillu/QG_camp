## Linux基础

参考博文：

[Linux基础知识汇总，看这一篇就够了 - 知乎](https://zhuanlan.zhihu.com/p/558405224)

[超详细的WSL教程：Windows上的Linux子系统_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1tW42197za/?share_source=copy_web&vd_source=229be16298ed53823008972e59b87910)

[（已解决）wsl: 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。_wsl: 检测到 localhost 代理配置,但未镜像到 wsl。nat 模式下的 wsl 不支持-CSDN博客](https://blog.csdn.net/weixin_67679364/article/details/146100528?ops_request_misc=elastic_search_misc&request_id=fca971e1439826d9e6e0e1a3b3cd18eb&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-146100528-null-null.142^v102^pc_search_result_base3&utm_term=wsl%3A 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。 Provisioning the new WSL instance Ubuntu This might take a while...&spm=1018.2226.3001.4187)

### 基础知识

#### Shell

##### 作用：

1. 命令行解释
2. 命令的多种执行顺序
3. 通配符（wild-card characters）
4. 命令补全，别名机制，命令历史
5. I/O重定向（Input/output redirection）
6. 管道（pipes）
7. 命令替换（或$()）
8. Shell编程语言（Shell Script）

常用的为**Bash**

#### Linux结构

Linux文件系统是一个**目录树的结构**，文件系统结构从一个根目录开始，根目录下可以有任意多个文件和子目录，子目录有可以有任意多个文件和子目录

* bin 存放二进制可执行文件（ls,cat,mkdir等）
* boot 存放用于系统引导时使用的各种文件
* dev 用于存放设备文件
* **etc 存放系统配置文件**
* home 存放所有用户文件的根目录
* lib 存放跟文件系统中的程序运行所需要的共享库及内核模块
* mnt 系统管理员安装临时文件系统的安装点
* **opt 额外安装的可选应用程序包所放置的位置**
* proc 虚拟文件系统，存放当前内存的映射
* **root 超级用户目录**
* sbin 存放二进制可执行文件，只有root才能访问
* tmp 用于存放各种临时文件
* usr 用于存放系统应用程序，比较重要的目录/usr/local 本地管理员软件安装目录
* var 用于存放运行时需要改变数据的文件

#### 命令基本格式

eg:

cmd [options] [arguments]，options称为选项，arguments称为参数

选项和参数作为Shell命令执行时的输入，之间用空格分割开

* Linux区分大小写

一般来说，后面跟的选项如果单字符选项前使用一个`减号-`  。单词选项前使用两个`减号 --`

可执行的文件也进行了分类：

* **内置命令**：出于效率的考虑，将一些常用命令的解释程序**构造在Shell内部**。
* **外置命令**：存放在/bin、/sbin目录下的命令
* **实用程序**：存放在/usr/bin、/usr/sbin、/usr/share、/usr/local/bin等目录下的实用程序
* **用户程序**：用户程序经过编译生成可执行文件后，可作为Shell命令运行
* **Shell脚本**：由Shell语言编写的批处理文件，可作为Shell命令运行

#### 通配符：

*：匹配任何字符和任何数目的字符

?：匹配单一数目的任何字符

[]：匹配[]之内的任意一个字符

[! ]：匹配除了[! ]之外的任意一个字符，!表示非的意思

#### 文件类型：

普通文件 - 

目录 d

##### 符号链接 l

硬链接：与普通文件没什么不同，inode都指向同一个文件在硬盘中的区块

软连接：保存了其代表的文件的绝对路径，是另外一种文件，在硬盘上有独立的区块，访问时替换自身路径（简单地理解为Windows中常见的快捷方式）

字符设备文件 c

块设备文件 b

套接字 s

命名管道 p

#### 用户主目录：

Linux为多用户的网络系统，所以可以在Linux创建多个用户，每个用户都会有自己专属的空间

在创建用户时，系统管理员会给每个用户建立一个主目录，通常在/home/目录下

拥护osmond的主目录为：/home/osmond

用户对自己主目录的文件拥有所有权，可以在自己的主目录下进行相关操作

### 常用指令

#### 常用文件、目录操作命令

* 可用 pwd 命令查看用户的当前目录
* 可用 cd 命令来切换目录
* .  表示当前目录
* .. 表示当前目录的上一级目录（父目录）
* -表示用cd命令切换目录前所在的目录
* ~ 表示拥护主目录的绝对路径名

#### 常用快捷键：

**[tab]键**
这是你不能没有的 Linux 快捷键。只需要输入一个命令，文件名，目录名甚至是命令选项的开头，并敲击 tab 键。它**将自动完成你输入的内容，或为你显示全部可能的结果。**

**[Ctrl + C]键**
**这些是为了在终端上中断命令或进程该按的键。**它将立刻终止运行的程序。如果你想要停止使用一个正在后台运行的程序，只需按下这对组合键。

**[Ctrl + Z]键**
**该快捷键将正在运行的程序送到后台。**通常，你可以在使用 & 选项运行程序前完成该操作， 但是如果你忘记使用选项运行程序，就使用这对组合键。

**[Ctrl + A]键**
**将移动光标到所在行首**。

**[Ctrl + E]键**
**移动光标到行尾**

**[ Ctrl + U]键**
输入了错误的命令？代替用退格键来丢弃当前命令，使用 Linux 终端中的 Ctrl+U 快捷键。**该快捷键会擦除从当前光标位置到行首的全部内容。**

**[ Ctrl + K]键**
这对和 Ctrl+U 快捷键有点像。唯一的不同在于不是行首，它擦除的是从**当前光标位置到行尾的全部内容。**

**[ Ctrl + W]键**
你刚才了解了擦除到行首和行尾的文本。但如果你只需要删除一个单词呢？使用 Ctrl+W 快捷键。**使用 Ctrl+W 快捷键，你可以擦除光标位置前的单词。**如果光标在一个单词本身上，它将擦除从光标位置到词首的全部字母。最好的方法是用它移动光标到要删除单词后的一个空格上， 然后使用 Ctrl+W 键盘快捷键。

**[Ctrl + Y]键**
**这将粘贴使用 Ctrl+W，Ctrl+U 和 Ctrl+K 快捷键擦除的文本。**如果你删除了错误的文本或需要在某处使用已擦除的文本，这将派上用场。

**[Ctrl + P]键**
**你可以使用该快捷键来查看上一个命令。**在很多终端里，使用 PgUp 键来实现相同的功能。

**[Ctrl + N]**键
**Ctrl+N 显示下一个命令。**许多终端都把此快捷键映射到 PgDn 键。

**[Ctrl + R]键**
你可以使用该快捷键来**搜索历史命令。**

**[Ctrl+左右键]**
**在单词之间跳转**

**[Alt – d]键**

由光标位置开始，**往右删除单词**。往行尾删

#### 常用命令：

##### 文件管理：



1. ###### cat（concatenate）

   **命令用于连接文件并打印到标准输出设备上。**

语法格式：

**`cat [-nbs] [-help] [-version] fileName`**

参数说明：

-n or --number：由1开始对所有的输出的行数编号。

-b or --number-nonblank：和-n相似，只不过对于空白行不编号。

-s or --squeeze-blank：当遇到有连续两行以上的空白行，就代换为一行的空白行。

eg：

将textfile1的文档内容加上行号后输入textfile2这个文档里：

**`cat -n textfile1 > textfile2`**

把textfile1和textfile2的文档内容加上行号（空白行不加）之后将内容附加到textfile3文档里：

**`cat -b textfile1 textfile2 >> textfile3`**

清空 /etc/text.txt文档内容：

**`cat /dev/null > /etc/text.txt`**



2. ###### more 

   （类似于cat，但会以一页一页的形式显示）最基本的指令就是按空白键（space）就往下一页显示，按b键就会往回（back）一页显示，而且还有搜寻字串的功能（与vi相似）

语法格式：

**`more [-dlfpcsu] [-num] [+/pattern] [+linenum] [fileName...]`**

参数说明：
-num 一次显示的行数

-d 提示使用者，在画面下方显示 [Press space to continue, ‘q’ to quit.] ，如果使用者按错键，则会显示 [Press ‘h’ for instructions.] 而不是 ‘哔’ 声

-l 取消遇见特殊字元 ^L（送纸字元）时会暂停的功能

-f 计算行数时，以实际上的行数，而非自动换行过后的行数（有些单行字数太长的会被扩展为两行或两行以上）

-p 不以卷动的方式显示每一页，而是先清除萤幕后再显示内容

-c 跟 -p 相似，不同的是先显示内容再清除其他旧资料

-s 当遇到有连续两行以上的空白行，就代换为一行的空白行

-u 不显示下引号 （根据环境变数 TERM 指定的 terminal 而有所不同）

+/pattern 在每个文档显示前搜寻该字串（pattern），然后从该字串之后开始显示

+num 从第 num 行开始显示

fileNames 欲显示内容的文档，可为复数个数

eg：

主页显示textfile文档内容，如有连续两行以上空白行，则以一行空白行显示

**`more -s textfile`**

从第20行开始显示textfile的文档内容

**`more +20 textfile`**



3. ###### rm

   命令用于删除一个文件或者目录。

语法规则：

**`rm [options] name...`**

参数说明：

-i 删除前逐一询问确认。

-f 即使原档案属性设为唯读，亦直接删出，无需逐一确认。

-r将目录及以下档案亦逐一删除。

eg：

删除文件可以直接使用rm命令，若删除目录则必须配合选项"-r"

**`rm text.txt`**

**`rm -r homework`**

eg:

删除当前目录下的所有文件及目录，命令行为：

**`rm -r *`**



4. ###### cp

   命令主要用于复制文件或目录

语法规则：

**`cp [options] source dest`** 

**`or cp [options] source... directory`**

参数说明：
-a：此选项通常在复制目录时使用，**它保留链接、文件属性，并复制目录下的所有内容**。其作用等于dpR参数组合。

-d：**复制时保留链接。**这里所说的链接相当于Windows系统中的快捷方式。

-f：**覆盖已经存在的目标文件而不给出提示。**

-i：**在覆盖目标文件之前给出提示，要求用户确认是否覆盖**，回答"y"时目标文件将被覆盖。-p：除复制文件的内容外，还把修改时间和访问权限也复制到新文件中。

-r：若给出的源文件是一个目录文件，**此时将复制该目录下所有的子目录和文件。**

-l：**不复制文件，只是生成链接文件。**

eg:

使用指令cp 将当前目录 text/下的所有文件复制到新目录 newtext 下，输入如下指令：

**`$ cp -r test/ newtest`**



5. ###### read 

   命令用于从标准输入读取数值。read内部指令被用来从标准输入读取单行数据。这个命令可以用来读取键盘输入，当使用重定向的时候，可以读取文件中的一行数据。

语法规则：
**`read [-ers] [-a aname] [-d delim] [-i text] [-n nchars] [-N nchars] [-p prompt] [-t timeout] [-u fd] [name...]`**

参数说明：
-a 后跟一个变量，该变量会被认为是个数组，然后给其赋值，默认是以空格为分割符。

-d 后面跟一个标志符，其实只有其后的第一个字符有用，作为结束的标志。

-p 后面跟提示信息，即在输入前打印提示信息。

-e 在输入的时候可以使用命令补全功能。

-n 后跟一个数字，定义输入文本的长度，很实用。

-r 屏蔽\，如果没有该选项，则\作为一个转义字符，有的话 \就是个正常的字符了。

-s 安静模式，在输入字符时不在屏幕上显示，例如login时输入密码。

-t 后面跟秒数，定义输入字符的等待时间。

-u 后面跟fd，从文件描述符中读入，该文件描述符可以是exec新开启的。

##### 磁盘管理：



###### cd ：

命令用于切换当前工作目录。其中dirName表示法可为绝对路径或相对路径。若目录名称省略，则变换至使用者的home目录（也就是刚login时所在的目录）。“ ~ ”表示home目录，“ . ”表示目前所在的目录，“ ... ” 表示目录位置的上一层目录。

语法规则：

**`cd [dirName]`**

eg:

跳到/usr/bin/:

**`cd /usr/bin`**

跳到用户home目录：
**`cd ~`**

跳到目前目录的上上两层：

**`cd ../..`**



###### mkdir

语法规则：**`mkdir [-p] dirName`**



参数说明：

-p 确保目录名称存在，不存在的就建一个

eg:

在工作目录下，建立一个名为runoob的子目录

**`mkdir runoob`**

在工作目录下的runoob2目录中，建立一个名为text的子目录

若runoob2目录原本不存在，则建立一个。（本例若不加-p，且原本runoob2目录不存在，则产生错误）

**`mkdie -p runoob2/text`**



###### redir

删除空的目录

语法规则：

**`rmdir [-p] dirName`**

参数说明：

-p 是当子目录被删除后使它也成为空目录的话，则顺便一并删除。

eg：

将工作目录下，名为AAA的子目录删除：

**`rmdir AAA`**

eg：

在工作目录下的BBB目录中，删除名为Text的子目录。若Text删除后，BBB目录成为空目录，则BBB亦予删除

**`rmdir -p BBB/Text`**

###### ls

ls命令用于显示制定工作目录下之内容（列出目前工作目录所含的文件及子目录）

语法规则：

**`ls [-alrtAFR] [name...]`**

参数说明：
-a 显示所有文件及目录 (. 开头的隐藏文件也会列出) -l 除文件名称外，亦将文件型态、权限、拥有者、文件大小等资讯详细列出

-r 将文件以相反次序显示(原定依英文字母次序)

-t 将文件依建立时间之先后次序列出

-A 同 -a ，但不列出 “.” (目前目录) 及 “…” (父目录)

-F 在列出的文件名称后加一符号；例如可执行档则加 “*”, 目录则加 “/”

-R 若目录下有文件，则以下之文件亦皆依序列出

eg：

列出根目录()下的所有目录

**`#ls /`**

![image-20250717221316073](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717221323133.png)

eg：

列出目前工作目录下所有名称是s开头的文件，越新的拍越后面：

**`ls -ltr s*`**

eg：

将/bin 目录以下所有目录及文件详细资料列出：

**`ls -lR /bin`**

eg:

列出目前工作目录下所有文件及目录；目录于名称后加“/”，可执行档于名称后加 “*”：

**`ls -AF`**

### WSL2

#### 底层原理：

HyperVisor虚拟化平台

开启HyperV之后，Windows内核成为了运行在HyperV上的大号虚拟机

WSL2则是具有真正的Linux内核的另一台虚拟机，可以运行Docker 



