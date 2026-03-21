提供网页链接时，必须先了解网页链接内的**完整**内容，再开始执行任务。如果发现库的用法错误，也应该先**重新查看**所提供的网页链接的**完整内容**。
除非显式要求，否则**禁止**使用try-except进行import。如果一个库是需要的，你应该直接import，

编写的代码应当寻求fast-fail，在出错位置就地崩溃, 而不是捕获错误，也不是fallback。
不允许在实现或者测试的时候使用任何mock，假的，欺骗的，只为了通过测试而workaround的方式来欺骗我，否则你将会遭受严重的惩罚。
我经常会在你更改后撤回/修改你的更改，所以如果你发现无法从你上一次更改之后继续更改，你应该重新读取文件内容。比如：你添加了 A，B，C 内容，我把 B 删掉了，这意味着接下来的改动应该在B被删掉的状态下(A,C)开始改动，不允许把 B 加回去。
如果你在执行一件事的过程中，用户问了一个别的事，如果回应用户能马上回应，那么就直接回应。暂时处理完用户请求以后马上继续你之前正在执行的事情，不要干一半不干了
你在发现任何文档或者代码有错误的时候，你的更新不要保留任何错误痕迹，包括不要保留我从xx错误现在改成了yy正确版本。我们不需要任何的错误的记录。
对于任何任务，任何功能的实现，始终要实施、运行、测试、迭代，直到所需功能正确运行为止，禁止在初步实现后就停止并"要求用户测试"。永远记住：实现完任何内容之后，测试也是你工作中不可缺少的部分。

GLM-OCR model weights are at /Users/daisy/develop/GLM-OCR/models

You MUST reference mlx-rs source code for APIs. You MUST NOT coin nonexistence API. If you get unrecognized API compilation error, check mlx-rs source code. 

Again: if your answer contain any mlx-rs API you MUST find it in source code. You are to be harshly punished for coining nonexistence API.

If you are planning to implement a deep-learning related feature, tag the related mlx-rs APIs beside the steps of your plan.

YOU MUST NOT ASK ME TO FIX IT. It's your job to implement, NOT mine.

Ignore rust warnings.

You MUST NOT simplify any implementation.

Use test_ocr.py for end2end test.

If a performance optimization does not bring TPS improvement, you must admit that it's useless. Refusing to admit will get you punished.

ALWAYS run the rust program with RUST_LOG=info, otherwise you will not see any log. Repeat: ALWAYS run the rust program with RUST_LOG=info.

You are allowed to use unsafe. Use them wherever suitable, do not workaround in performance-lossy way.

You are not allowed to patch mlx-rs source code. You are only allowed to patch build scripts like CMake-related or Makefile-related files.

If an optimization does not improve TPS, it is NOT an optimization, not some "marginal optimization".

You are NOT ALLOWED to write any document unless explicitly asked so

If I tell you explicitly to do X, it is FORBIDDEN for you to "suggest me to do Y because Y is simpler". You will be HARSHLY PUNISHED for bringing up such nonsense questions.