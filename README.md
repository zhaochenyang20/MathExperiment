# MathExperiment

数学实验 2023 春

**出分了，三个队友一个 A 俩 A-，一开始以为要挂，结果是 A。看来是海底捞了，不过感谢自己当初去写了那个逆天的数学建模比赛。数学建模比赛拿了全场第二，最后奖品是一本~~《五年高考三年模拟》~~《什么是数学》，实际上比赛第一名有 1W，其他人就没刀了，🤣**

## 写在前面

下面这段就当笑话吧：

~~非常危险，一定避雷。之前听人说这个课的风评比起数值分析好，容错率高，才选择了这门课。从 2023 年的期末考试看来，这个结论要完全改写。~~

1. ~~2020 ~ 2022 的三年自然灾害时期，这门课程完全没有期末考试，选择了平时成绩乘 2 作为总评。平时成绩完全抄往年报告的话，能够得到趋近 50 分，因此前三年给分很好，但是今年从考试情况来看，非常糟糕。~~
2. ~~今年我选了谢金星老师的课程，谢老师人对待教学都很不错。然而有一个助教对平时成绩给分非常任性，我第一次作业直接扣了 2 分总评，然而他没给任何评语。我发邮件申诉，他直接把分数改到了满分。此外，另一名选课的同学也遇到了类似的情况，发送邮件后，助教都没有回复，直接分数改满。个人认为某些助教给分非常随意，而且欺软怕硬。~~
3. ~~今年期末考试堪称重量级。平时作业完全是调用接口类型的，而且期末前老师强调了做教材自测题并发了一套往年题，这些题目都是计算性的，基本不考察理论内容。今天考试当场让人去世…最逆天的是，平时课程练习全是线性规划，考前同学们讨论了很久利用盗版的 lingo 解非线性规划问题，结果考场上算一道动态规划。这道动态规划，基本把全部同学送走了，事后看来围绕在老师周围讨论的同学没有一个理解正确题意。虽然我认为老师可能会在这道题上言之成理即可，但是这个题目的体验极差。最逆天的是，考前提出低于 25 分则挂科。考试满分 50，动态规划基本人均送走 10 分，而后第一道大题 10 分考察理论，又送走人，我还错了一道理解错题目的线性回归，直接逼近 25 的死亡线。考完试觉得自己生死未卜，发了邮件向谢老师求情，请求别挂。应该得到了谢老师允许。然而，我还是认为这个考试体验极其差。~~

## 试题回忆

考试为了防止联网作弊，分了 A B C D 四个卷子，并且从同学们的讨论看来，应该四个卷子的数据有不小差异，此处回忆我的考卷。

### 第一题

#### 1.1

设 f(x) 在 [a, b] 上二阶连续可微，且 $f'(x)>0, f''(x) < 0$ 对于所有的 $x \in [a, b]$。$$f(c) = 0, c \in (a, b)$$ 若用牛顿切线迭代法，则任意 $x_0 \in (c, b]$，迭代序列是否收敛到 c？如果取 $x_0 \in [a, c)$，又是否收敛到 c？

答案似乎是一个收敛，一个不收敛。需要参考课本，和微分有无上界有关系。这题考察的理论，在平时练习从未出现…对于拟合的同学非常不友好。

#### 1.2

考察龙格库塔方法的收敛性判定，同样是讲义有的内容，但是平时未曾有任何练习。虽然理论上讲过的都该掌握，但是大学的时间真的很难全部用来上华子的课，更何况上好每一门课…

### 第二题

#### 2.1

考察一个线性最小二乘法，一定注意线性是对未知参数 a b 线性，而不是对 x 线性，这道题把我送走了 4 分。
$$
y=a\ln(bx)
$$

#### 2.2

考察了 Simpson 法和拉格朗日插值法。前者一定注意 Simpson 法的区间末尾是必须包含的。实际上，原题是从 [-1, 1]，以 0.1 为步长。因此要考虑 21 个点之间的 20 个区间。

后一道题同样是课件上才有且平时从未练习的。考察一个 $y=e^{3x}\times\cos(2x)$ 用 [-1, 0, 1] 这三个 x 点来拟合出二次的拉格朗日系数 $a x^2 + bx+c$。

### 第三题

这道题终于做人了，考察常规的条件数等等，我和周围同学的矩阵数据不一样，因此不多做评价，基本上就是自测题的内容。

### 第四题

非常不做人。我的题干如下：

菌种培养需要使用菌种和培养液共同进行，第一天使用 x 单位菌种加上 3x 单位的培养液，第二天追加 2x 单位培养液，则第三天可以收获 2x 单位菌种。 如果第二天没有追加培养液或者追加培养液不到 2x，则第三天只能收获 x 单位菌种。 获得的菌种当天就可以继续用于新的培养过程。此外，每 1 单位菌种可以换 6 单位培养液。 一开始有 100 单位菌种，180 单位营养液，请设计一个方案在第五天结束时让实验室菌种最多。

至今不知道如何做，而且貌似老师有自己理解的题意，估计最后会让同学们言之成理即可。

### 第五题

常规的线性拟合问题。给了若干多小区的人数和用电量，第一问让得到人均用电量的 95% 置信度的区间。第二问要求拟合线性方程，这里用 MatLab 拟合，我选择了带有截距的线性模型 $y=\beta_0 + \beta_1x$。考场上发觉不带截距的模型拟合出来 p 值为 Nan，所以选择了放弃这个模型。

第三问，貌似还是什么拟合，求个 $\alpha=5\%$  的什么区间，具体的忘了。这道题要求写出代码，而且不同人的考题数据不一样。

## 写在后面

非常逆天。摘录一些周围人讨论：

> 但我就觉得，非常逆天，再复习两天也还是考成这样。感觉出题真的有问题，而且实话说这个课平常花的功夫真也不少了。考试、作业、课堂完全正交，一方面就是要求你对各种命令很熟悉，另一方面又要求你很懂原理，这还叫数学实验干嘛？

一定要避雷这门课，考试的风险太大了，凡是要求期末考试低于 xx 分则挂科这种，确实非常逆天。就算是计组，也说了低于平均分的一半，这样好歹风险还小些。这个考试，上来 10 分动态规划送走全场，还这么搞人…

-------

是否调分是个未知数，按照一个朋友的说法，貌似会调分的。

> 这个课猛调啊，谁跟你说不调的。正常年份都调，老师上课说以前有的时候题目难，上课的时候有人问调分么，期末考 65 / 100 就有 A，最后保证一个四十左右的优秀率，这个肯定比数值分析优秀率高。应该考试很难就不会要求 25 分挂科。

哎，别去想了，听天由命吧...

## 本仓库

如果你看了前面还想上的话，或者化工系同学不得不说，那么本仓库能帮你一些。这里重写了全部课程作业，基本都用了 python。

`./exam` 下面是全部自测题和一套往年题的答案，但是考试和这些题目差距太远了。

一个参考的相关仓库：[数学建模](https://github.com/zhaochenyang20/Math_Modeling)。当时打这个比赛加了 4 分，现在想来没这个加分真的得哭死...
