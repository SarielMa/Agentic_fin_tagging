# 代码冻结 — 2026-07-30 03:11,直到这一批 26 个 job 排完

用户的判断(2026-07-30):"每隔几个小时你找出几个错、改了、再跑,这个循环已经持续一周了……
你这么一改,其他还在跑的东西和你新的代码还是不是一个代码?"

**规则**:在 `ask6_batch_20260730_jobids.txt` 里的 26 个 job 全部结束之前,不修改
`data_whole_pipeline/` 里任何 `.py` / `.sh`,**包括注释**(pending job 是起跑时才读代码的,
改注释虽不改行为,但会让"同一份代码"这句话变得需要解释)。

允许的动作:读、跑 CPU 分析脚本、改 `paper/` 里的 tex 与 md、写 `_superseded_20260730/` 之类的
说明文件。

发现新问题怎么办:**记进 `TRACK1_ASK6_CHECKLIST.md` 的"下一批"一节,不要动手。** 判据:
除非它会让这 26 个 job 产出**无法使用**的结果(不是"不够完美"),否则等这批落地。

冻结时的配置断言(全部 ok,见提交脚本 phase 0):
verifier 问=算=六维;beta 0.6;K_v 10;K 200;unjudged fill mean;fusion sum;scaling range;
J 2;w_cov 1.0;window_source fused;verdict token cap 2816;baseline token cap 2048;
llmonly_* 臂不再传 --top-m。
