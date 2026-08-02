# Autopilot auto-log (machine-generated; prose lives in docs/RESULTS.md)

Updated: 2026-08-02T09:11:42Z — last milestone: 7B transfer + references (budget 2048)

## Step failures
```
2026-07-31T08:32:25Z steering-acceptance rc=1
[2026-08-01T02:35Z] RESOLVED MANUALLY: steering-acceptance rerun after fixing the tuple-unpack crash + re-grading both arms. Verdict REJECTED, D6 exit-led. See runs/r1_qwen_1p5b/analysis/steering_acceptance.json and commit 6e7936f.
```

## Banked results (all acc= lines, all logs, deduped)
```
run_baselines[budget_prompt 05b4b767ba]: gsm8k/dev acc=0.935 mean_think=1368
run_baselines[budget_prompt 05b4b767ba]: math_train/dev acc=0.838 mean_think=4469
run_baselines[budget_prompt 0ffacaf1d3]: gsm8k/dev acc=0.930 mean_think=1266
run_baselines[budget_prompt 0ffacaf1d3]: math_train/dev acc=0.855 mean_think=4439
run_baselines[budget_prompt 2d1d25d9dc]: gsm8k/dev acc=0.925 mean_think=1351
run_baselines[budget_prompt 2d1d25d9dc]: math_train/dev acc=0.859 mean_think=4411
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.133 mean_think=15423
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.233 mean_think=14166
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.233 mean_think=14686
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.267 mean_think=13011
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.267 mean_think=14653
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.300 mean_think=13752
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.300 mean_think=14056
run_baselines[budget_prompt ce42341dc5]: aime/test acc=0.333 mean_think=12801
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.212 mean_think=7928
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.227 mean_think=8199
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.232 mean_think=8514
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.242 mean_think=8196
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.258 mean_think=8054
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.268 mean_think=7683
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.273 mean_think=7628
run_baselines[budget_prompt ce42341dc5]: gpqa_diamond/test acc=0.308 mean_think=7916
run_baselines[budget_prompt ce42341dc5]: gsm8k/dev acc=0.935 mean_think=1156
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.840 mean_think=2047
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.844 mean_think=1979
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.848 mean_think=1982
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.872 mean_think=1943
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.872 mean_think=2007
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.876 mean_think=1830
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.884 mean_think=1700
run_baselines[budget_prompt ce42341dc5]: gsm8k/test acc=0.896 mean_think=1838
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.824 mean_think=4478
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.826 mean_think=4421
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.826 mean_think=4552
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.828 mean_think=4562
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.834 mean_think=4396
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.836 mean_think=4498
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.840 mean_think=4663
run_baselines[budget_prompt ce42341dc5]: math500/test acc=0.842 mean_think=4541
run_baselines[budget_prompt ce42341dc5]: math_train/dev acc=0.858 mean_think=4431
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.167 mean_think=13505
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.200 mean_think=13824
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.200 mean_think=15679
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.267 mean_think=13425
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.267 mean_think=13969
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.300 mean_think=14456
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.300 mean_think=15053
run_baselines[concise_prompt 5a781303f9]: aime/test acc=0.333 mean_think=12810
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.258 mean_think=7059
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.278 mean_think=6205
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.278 mean_think=6512
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.293 mean_think=6552
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.298 mean_think=6469
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.303 mean_think=6279
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.323 mean_think=6946
run_baselines[concise_prompt 5a781303f9]: gpqa_diamond/test acc=0.338 mean_think=7517
run_baselines[concise_prompt 5a781303f9]: gsm8k/dev acc=0.865 mean_think=179
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.752 mean_think=397
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.760 mean_think=342
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.760 mean_think=343
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.760 mean_think=348
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.764 mean_think=357
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.768 mean_think=395
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.772 mean_think=318
run_baselines[concise_prompt 5a781303f9]: gsm8k/test acc=0.784 mean_think=252
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.820 mean_think=3486
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.822 mean_think=3561
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.824 mean_think=3338
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.826 mean_think=3456
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.830 mean_think=3611
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.844 mean_think=3321
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.844 mean_think=3515
run_baselines[concise_prompt 5a781303f9]: math500/test acc=0.848 mean_think=3286
run_baselines[concise_prompt 5a781303f9]: math_train/dev acc=0.839 mean_think=3523
run_baselines[noop 2217fed51d]: aime/test acc=0.167 mean_think=14370
run_baselines[noop 2217fed51d]: aime/test acc=0.167 mean_think=15348
run_baselines[noop 2217fed51d]: aime/test acc=0.200 mean_think=15564
run_baselines[noop 2217fed51d]: aime/test acc=0.233 mean_think=13236
run_baselines[noop 2217fed51d]: aime/test acc=0.233 mean_think=13838
run_baselines[noop 2217fed51d]: aime/test acc=0.233 mean_think=15365
run_baselines[noop 2217fed51d]: aime/test acc=0.267 mean_think=14812
run_baselines[noop 2217fed51d]: aime/test acc=0.300 mean_think=14297
run_baselines[noop 2217fed51d]: aime/test acc=0.367 mean_think=13119
run_baselines[noop 2217fed51d]: aime/test acc=0.400 mean_think=12261
run_baselines[noop 2217fed51d]: aime/test acc=0.400 mean_think=13266
run_baselines[noop 2217fed51d]: aime/test acc=0.400 mean_think=14070
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.207 mean_think=7785
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.212 mean_think=7725
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.232 mean_think=8035
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.237 mean_think=7777
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.237 mean_think=7921
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.253 mean_think=8534
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.263 mean_think=8366
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.278 mean_think=8075
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.364 mean_think=6432
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.364 mean_think=6610
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.394 mean_think=6632
run_baselines[noop 2217fed51d]: gpqa_diamond/test acc=0.404 mean_think=6055
run_baselines[noop 2217fed51d]: gsm8k/dev acc=0.960 mean_think=1308
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.856 mean_think=2089
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.864 mean_think=2065
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.868 mean_think=2094
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.868 mean_think=2110
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.876 mean_think=2057
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.888 mean_think=2108
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.892 mean_think=2009
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.900 mean_think=2122
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.908 mean_think=1480
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.928 mean_think=1264
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.928 mean_think=1397
run_baselines[noop 2217fed51d]: gsm8k/test acc=0.928 mean_think=1526
run_baselines[noop 2217fed51d]: math500/test acc=0.824 mean_think=4780
run_baselines[noop 2217fed51d]: math500/test acc=0.826 mean_think=4563
run_baselines[noop 2217fed51d]: math500/test acc=0.828 mean_think=4423
run_baselines[noop 2217fed51d]: math500/test acc=0.830 mean_think=4668
run_baselines[noop 2217fed51d]: math500/test acc=0.832 mean_think=4727
run_baselines[noop 2217fed51d]: math500/test acc=0.836 mean_think=4620
run_baselines[noop 2217fed51d]: math500/test acc=0.840 mean_think=4812
run_baselines[noop 2217fed51d]: math500/test acc=0.844 mean_think=4629
run_baselines[noop 2217fed51d]: math500/test acc=0.922 mean_think=3428
run_baselines[noop 2217fed51d]: math500/test acc=0.924 mean_think=3435
run_baselines[noop 2217fed51d]: math500/test acc=0.924 mean_think=3572
run_baselines[noop 2217fed51d]: math500/test acc=0.924 mean_think=3757
run_baselines[noop 2217fed51d]: math_train/dev acc=0.865 mean_think=4529
run_baselines[static_budget 0047e49359]: gsm8k/dev acc=0.945 mean_think=1311
run_baselines[static_budget 0047e49359]: math_train/dev acc=0.719 mean_think=3536
run_baselines[static_budget 974b1c9039]: aime/test acc=0.100 mean_think=2021
run_baselines[static_budget 974b1c9039]: aime/test acc=0.100 mean_think=2034
run_baselines[static_budget 974b1c9039]: aime/test acc=0.100 mean_think=2047
run_baselines[static_budget 974b1c9039]: aime/test acc=0.133 mean_think=2014
run_baselines[static_budget 974b1c9039]: aime/test acc=0.167 mean_think=2022
run_baselines[static_budget 974b1c9039]: aime/test acc=0.167 mean_think=2048
run_baselines[static_budget 974b1c9039]: aime/test acc=0.200 mean_think=2033
run_baselines[static_budget 974b1c9039]: aime/test acc=0.200 mean_think=2042
run_baselines[static_budget 974b1c9039]: aime/test acc=0.200 mean_think=2045
run_baselines[static_budget 974b1c9039]: aime/test acc=0.200 mean_think=2046
run_baselines[static_budget 974b1c9039]: aime/test acc=0.233 mean_think=2033
run_baselines[static_budget 974b1c9039]: aime/test acc=0.233 mean_think=2036
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.278 mean_think=1926
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.283 mean_think=1915
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.298 mean_think=1935
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.313 mean_think=1944
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.328 mean_think=1896
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.328 mean_think=1920
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.338 mean_think=1934
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.348 mean_think=1908
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.359 mean_think=1915
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.404 mean_think=1895
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.419 mean_think=1924
run_baselines[static_budget 974b1c9039]: gpqa_diamond/test acc=0.449 mean_think=1936
run_baselines[static_budget 974b1c9039]: gsm8k/dev acc=0.875 mean_think=1085
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.836 mean_think=1278
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.840 mean_think=1263
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.848 mean_think=1290
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.860 mean_think=1310
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.864 mean_think=1244
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.868 mean_think=1286
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.884 mean_think=1285
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.884 mean_think=1296
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.892 mean_think=1155
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.916 mean_think=1140
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.924 mean_think=1124
run_baselines[static_budget 974b1c9039]: gsm8k/test acc=0.932 mean_think=1115
run_baselines[static_budget 974b1c9039]: math500/test acc=0.758 mean_think=1731
run_baselines[static_budget 974b1c9039]: math500/test acc=0.762 mean_think=1704
run_baselines[static_budget 974b1c9039]: math500/test acc=0.768 mean_think=1718
run_baselines[static_budget 974b1c9039]: math500/test acc=0.772 mean_think=1731
run_baselines[static_budget 974b1c9039]: math500/test acc=0.776 mean_think=1703
run_baselines[static_budget 974b1c9039]: math500/test acc=0.776 mean_think=1747
run_baselines[static_budget 974b1c9039]: math500/test acc=0.786 mean_think=1719
run_baselines[static_budget 974b1c9039]: math500/test acc=0.794 mean_think=1750
run_baselines[static_budget 974b1c9039]: math500/test acc=0.814 mean_think=1677
run_baselines[static_budget 974b1c9039]: math500/test acc=0.814 mean_think=1706
run_baselines[static_budget 974b1c9039]: math500/test acc=0.816 mean_think=1669
run_baselines[static_budget 974b1c9039]: math500/test acc=0.816 mean_think=1676
run_baselines[static_budget 974b1c9039]: math_train/dev acc=0.688 mean_think=1713
run_baselines[static_budget c2477df0fe]: gsm8k/dev acc=0.925 mean_think=1222
run_baselines[static_budget c2477df0fe]: math_train/dev acc=0.731 mean_think=2593
run_baselines[static_budget c906237a84]: gsm8k/dev acc=0.725 mean_think=843
run_baselines[static_budget c906237a84]: math_train/dev acc=0.524 mean_think=996
run_baselines[trial_decode be5e561033]: aime/test acc=0.000 mean_think=332
run_baselines[trial_decode be5e561033]: aime/test acc=0.033 mean_think=309
run_baselines[trial_decode be5e561033]: aime/test acc=0.033 mean_think=316
run_baselines[trial_decode be5e561033]: aime/test acc=0.033 mean_think=336
run_baselines[trial_decode be5e561033]: aime/test acc=0.033 mean_think=344
run_baselines[trial_decode be5e561033]: aime/test acc=0.067 mean_think=295
run_baselines[trial_decode be5e561033]: aime/test acc=0.067 mean_think=325
run_baselines[trial_decode be5e561033]: aime/test acc=0.067 mean_think=341
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.111 mean_think=380
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.116 mean_think=374
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.126 mean_think=365
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.131 mean_think=368
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.141 mean_think=372
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.146 mean_think=362
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.152 mean_think=393
run_baselines[trial_decode be5e561033]: gpqa_diamond/test acc=0.157 mean_think=391
run_baselines[trial_decode be5e561033]: gsm8k/dev acc=0.520 mean_think=292
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.552 mean_think=305
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.556 mean_think=307
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.568 mean_think=297
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.576 mean_think=295
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.588 mean_think=299
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.592 mean_think=297
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.612 mean_think=304
run_baselines[trial_decode be5e561033]: gsm8k/test acc=0.624 mean_think=302
run_baselines[trial_decode be5e561033]: math500/test acc=0.478 mean_think=321
run_baselines[trial_decode be5e561033]: math500/test acc=0.484 mean_think=315
run_baselines[trial_decode be5e561033]: math500/test acc=0.490 mean_think=321
run_baselines[trial_decode be5e561033]: math500/test acc=0.492 mean_think=318
run_baselines[trial_decode be5e561033]: math500/test acc=0.494 mean_think=310
run_baselines[trial_decode be5e561033]: math500/test acc=0.498 mean_think=315
run_baselines[trial_decode be5e561033]: math500/test acc=0.500 mean_think=325
run_baselines[trial_decode be5e561033]: math500/test acc=0.506 mean_think=309
run_baselines[trial_decode be5e561033]: math_train/dev acc=0.362 mean_think=320
run_controller[exit_only 079322feb6]: aime/test acc=0.167 mean_think=9725
run_controller[exit_only 079322feb6]: aime/test acc=0.200 mean_think=10209
run_controller[exit_only 079322feb6]: aime/test acc=0.200 mean_think=11350
run_controller[exit_only 079322feb6]: aime/test acc=0.200 mean_think=12065
run_controller[exit_only 079322feb6]: aime/test acc=0.200 mean_think=9926
run_controller[exit_only 079322feb6]: aime/test acc=0.233 mean_think=10631
run_controller[exit_only 079322feb6]: aime/test acc=0.233 mean_think=13492
run_controller[exit_only 079322feb6]: aime/test acc=0.233 mean_think=6770
run_controller[exit_only 079322feb6]: aime/test acc=0.267 mean_think=13061
run_controller[exit_only 079322feb6]: aime/test acc=0.300 mean_think=7940
run_controller[exit_only 079322feb6]: aime/test acc=0.333 mean_think=6784
run_controller[exit_only 079322feb6]: aime/test acc=0.367 mean_think=11200
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.242 mean_think=6555
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.288 mean_think=5974
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.298 mean_think=6622
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.338 mean_think=6364
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.338 mean_think=6603
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.338 mean_think=6863
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.369 mean_think=6575
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.374 mean_think=6748
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.414 mean_think=2591
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.429 mean_think=2580
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.444 mean_think=2555
run_controller[exit_only 079322feb6]: gpqa_diamond/test acc=0.444 mean_think=2701
run_controller[exit_only 079322feb6]: gsm8k/dev acc=0.590 mean_think=506
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.792 mean_think=810
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.828 mean_think=658
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.832 mean_think=673
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.832 mean_think=745
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.840 mean_think=777
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.844 mean_think=657
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.856 mean_think=589
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.864 mean_think=643
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.876 mean_think=454
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.896 mean_think=493
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.904 mean_think=500
run_controller[exit_only 079322feb6]: gsm8k/test acc=0.912 mean_think=487
run_controller[exit_only 079322feb6]: math500/test acc=0.790 mean_think=2301
run_controller[exit_only 079322feb6]: math500/test acc=0.798 mean_think=2401
run_controller[exit_only 079322feb6]: math500/test acc=0.804 mean_think=2186
run_controller[exit_only 079322feb6]: math500/test acc=0.804 mean_think=2285
run_controller[exit_only 079322feb6]: math500/test acc=0.806 mean_think=2253
run_controller[exit_only 079322feb6]: math500/test acc=0.808 mean_think=2392
run_controller[exit_only 079322feb6]: math500/test acc=0.822 mean_think=2328
run_controller[exit_only 079322feb6]: math500/test acc=0.822 mean_think=2349
run_controller[exit_only 079322feb6]: math500/test acc=0.832 mean_think=1400
run_controller[exit_only 079322feb6]: math500/test acc=0.836 mean_think=1401
run_controller[exit_only 079322feb6]: math500/test acc=0.836 mean_think=1464
run_controller[exit_only 079322feb6]: math500/test acc=0.878 mean_think=1327
run_controller[exit_only 079322feb6]: math_train/dev acc=0.682 mean_think=2260
run_controller[exit_only 42c3453b67]: gsm8k/dev acc=0.530 mean_think=698
run_controller[exit_only 42c3453b67]: math_train/dev acc=0.635 mean_think=3353
run_controller[exit_only 5733e5f65d]: gsm8k/dev acc=0.615 mean_think=719
run_controller[exit_only 5733e5f65d]: math_train/dev acc=0.639 mean_think=3387
run_controller[exit_only 7b873f923e]: gsm8k/dev acc=0.475 mean_think=1009
run_controller[exit_only 7b873f923e]: math_train/dev acc=0.600 mean_think=3960
run_controller[exit_only 7ea0180ba8]: gsm8k/dev acc=0.645 mean_think=521
run_controller[exit_only 7ea0180ba8]: math_train/dev acc=0.646 mean_think=2783
run_controller[exit_only 84bc53aa11]: gsm8k/dev acc=0.825 mean_think=501
run_controller[exit_only 84bc53aa11]: math_train/dev acc=0.828 mean_think=2223
run_controller[exit_only ab2ad2aa91]: gsm8k/dev acc=0.525 mean_think=559
run_controller[exit_only ab2ad2aa91]: math_train/dev acc=0.637 mean_think=2752
run_controller[exit_only cbb8895c16]: aime/test acc=0.133 mean_think=8620
run_controller[exit_only cbb8895c16]: aime/test acc=0.167 mean_think=8316
run_controller[exit_only cbb8895c16]: gpqa_diamond/test acc=0.121 mean_think=4744
run_controller[exit_only cbb8895c16]: gpqa_diamond/test acc=0.177 mean_think=4837
run_controller[exit_only cbb8895c16]: gsm8k/dev acc=0.680 mean_think=426
run_controller[exit_only cbb8895c16]: gsm8k/test acc=0.636 mean_think=512
run_controller[exit_only cbb8895c16]: gsm8k/test acc=0.672 mean_think=574
run_controller[exit_only cbb8895c16]: math500/test acc=0.624 mean_think=1528
run_controller[exit_only cbb8895c16]: math500/test acc=0.632 mean_think=1773
run_controller[exit_only cbb8895c16]: math_train/dev acc=0.660 mean_think=1707
run_controller[exit_only fdec51004e]: gsm8k/dev acc=0.665 mean_think=445
run_controller[exit_only fdec51004e]: math_train/dev acc=0.627 mean_think=2214
run_controller[full 33044f6502]: gsm8k/dev acc=0.605 mean_think=1659
run_controller[full 33044f6502]: math_train/dev acc=0.496 mean_think=6013
run_controller[full 499c0ea86f]: gsm8k/dev acc=0.530 mean_think=2077
run_controller[full 499c0ea86f]: math_train/dev acc=0.495 mean_think=6071
run_controller[full 60948c01b8]: gsm8k/dev acc=0.555 mean_think=2035
run_controller[full 60948c01b8]: math_train/dev acc=0.476 mean_think=6157
run_controller[full d20644a89a]: gsm8k/dev acc=0.635 mean_think=2319
run_controller[full d20644a89a]: math_train/dev acc=0.482 mean_think=5938
run_controller[full dac4beda09]: gsm8k/dev acc=0.615 mean_think=2351
run_controller[full dac4beda09]: gsm8k/dev acc=0.840 mean_think=717
run_controller[full dac4beda09]: math_train/dev acc=0.482 mean_think=5265
run_controller[full dac4beda09]: math_train/dev acc=0.606 mean_think=2045
run_controller[noop 5391bd1826]: aime/test acc=0.200 mean_think=15197
run_controller[noop 5391bd1826]: aime/test acc=0.233 mean_think=13098
run_controller[noop 5391bd1826]: aime/test acc=0.300 mean_think=14780
run_controller[noop 5391bd1826]: aime/test acc=0.333 mean_think=12450
run_controller[noop 5391bd1826]: gpqa_diamond/test acc=0.232 mean_think=7676
run_controller[noop 5391bd1826]: gpqa_diamond/test acc=0.258 mean_think=7508
run_controller[noop 5391bd1826]: gpqa_diamond/test acc=0.258 mean_think=8122
run_controller[noop 5391bd1826]: gpqa_diamond/test acc=0.384 mean_think=6573
run_controller[noop 5391bd1826]: gsm8k/dev acc=0.920 mean_think=1913
run_controller[noop 5391bd1826]: gsm8k/test acc=0.856 mean_think=2409
run_controller[noop 5391bd1826]: gsm8k/test acc=0.884 mean_think=2164
run_controller[noop 5391bd1826]: gsm8k/test acc=0.896 mean_think=1916
run_controller[noop 5391bd1826]: gsm8k/test acc=0.928 mean_think=1387
run_controller[noop 5391bd1826]: math500/test acc=0.834 mean_think=4491
run_controller[noop 5391bd1826]: math500/test acc=0.846 mean_think=4476
run_controller[noop 5391bd1826]: math500/test acc=0.852 mean_think=4535
run_controller[noop 5391bd1826]: math500/test acc=0.940 mean_think=3240
run_controller[noop 5391bd1826]: math_train/dev acc=0.855 mean_think=4663
run_controller[noop 888caea894]: gsm8k/dev acc=0.940 mean_think=1417
run_controller[noop 888caea894]: math_train/dev acc=0.701 mean_think=4470
run_controller[steer_only e3e61597cf]: gsm8k/dev acc=0.765 mean_think=3954
run_controller[steer_only e3e61597cf]: math_train/dev acc=0.463 mean_think=8631
select[budget_prompt]: 4 configs; best acc=0.873 (se 0.017); pick ce42341dc5 params={'tau_exit': np.float64(0.9), 'patience_k': np.int64(2), 'alpha': np.float64(6.0), 'budget': np.int64(1024)} acc=0.873 tokens=3776
select[exit_only]: 8 configs; best acc=0.664 (se 0.022); pick cbb8895c16 params={'tau_exit': np.float64(0.7), 'patience_k': np.int64(1), 'alpha': np.float64(6.0), 'budget': np.int64(4096)} acc=0.664 tokens=1451
select[static_budget]: 4 configs; best acc=0.770 (se 0.022); pick c2477df0fe params={'tau_exit': np.float64(0.9), 'patience_k': np.int64(2), 'alpha': np.float64(6.0), 'budget': np.int64(4096)} acc=0.770 tokens=2319
```

## runs/r1_qwen_1p5b/analysis/selection_dev.json
```json
{
  "families": {
    "exit_only": {
      "phash": "079322feb6",
      "acc": 0.817,
      "se": 0.018986467385353718,
      "tokens": 1908.876,
      "n_problems": 250,
      "params": {
        "tau_exit": 0.7,
        "patience_k": 2,
        "alpha": 6.0,
        "budget": 4096
      },
      "tuned_params": {
        "tau_exit": 0.7,
        "patience_k": 2
      },
      "one_se_threshold": 0.7980135326146462,
      "best_acc": 0.817,
      "best_se": 0.018986467385353718,
      "n_configs": 8
    },
    "full": {
      "phash": "dac4beda09",
      "acc": 0.62,
      "se": 0.023249994602063544,
      "tokens": 4682.456,
      "n_problems": 250,
      "params": {
        "tau_exit": 0.7,
        "patience_k": 1,
        "alpha": 6.0,
        "budget": 4096
      },
      "tuned_params": {
        "tau_exit": 0.7,
        "patience_k": 1
      },
      "one_se_threshold": 0.6188434656332817,
      "best_acc": 0.643,
      "best_se": 0.024156534366718288,
      "n_configs": 5
    },
    "static_budget": {
      "phash": "974b1c9039",
      "acc": 0.772,
      "se": 0.022850658862312204,
      "tokens": 1587.821,
      "n_problems": 250,
      "params": {
        "tau_exit": 0.9,
        "patience_k": 2,
        "alpha": 6.0,
        "budget": 2048
      },
      "tuned_params": {
        "budget": 2048
      },
      "one_se_threshold": 0.75536771340581,
      "best_acc": 0.777,
      "best_se": 0.02163228659419001,
      "n_configs": 4
    },
    "budget_prompt": {
      "phash": "ce42341dc5",
      "acc": 0.734,
      "se": 0.023690100960826164,
      "tokens": 3775.868,
      "n_problems": 250,
      "params": {
        "tau_exit": 0.9,
        "patience_k": 2,
        "alpha": 6.0,
        "budget": 1024
      },
      "tuned_params": {
        "budget": 1024
      },
      "one_se_threshold": 0.729063384002568,
      "best_acc": 0.752,
      "best_se": 0.022936615997432023,
      "n_configs": 4
    }
  },
  "accuracy_column": "correct_matched_strict",
  "source": "runs/r1_qwen_1p5b/analysis/regrade_records.parquet",
  "generated_utc": "2026-07-30T22:09:24Z"
}```

## runs/r1_qwen_1p5b/analysis/steering_acceptance.json
```json
{
  "alphas": {
    "3.0": {
      "phash": "e3e61597cf",
      "alpha": 3.0,
      "n_paired_problems": 250,
      "n_rollouts": 1000,
      "acceptance": {
        "delta": -0.215,
        "lo": -0.251,
        "hi": -0.18,
        "p_a_lt_b": 0.0,
        "p_a_gt_b": 1.0,
        "n_problems": 250,
        "max_plausible_drop": 0.251,
        "accepted": false
      },
      "tokens_delta": {
        "delta": 3582.619,
        "lo": 3130.73535,
        "hi": 4063.266725
      },
      "phase_shift_regex_paragraph_approx": {
        "verification": {
          "rate_steered": 0.0319,
          "rate_unsteered": 0.02525,
          "delta": 0.006649999999999996,
          "lo": 0.004423749999999998,
          "hi": 0.008926250000000002
        },
        "backtracking": {
          "rate_steered": 0.1631,
          "rate_unsteered": 0.23075,
          "delta": -0.06765000000000002,
          "lo": -0.07355874999999999,
          "hi": -0.06216249999999999
        },
        "deduction": {
          "rate_steered": 0.526,
          "rate_unsteered": 0.50005,
          "delta": 0.02595000000000003,
          "lo": 0.01992499999999997,
          "hi": 0.03292562499999997
        }
      }
    }
  },
  "any_accepted": false,
  "noop_reference": {
    "phash": "5391bd1826",
    "n_rollouts": 1000
  },
  "grading": "fixed extractor, answer budget matched at 512",
  "d6": {
    "noop_tokens": 4112.608,
    "exit_tokens": 1908.876,
    "exit_savings": 2203.732,
    "full_tokens": 4682.456,
    "full_acc": 0.62,
    "steer_extra_savings": -2773.58,
    "extra_over_exit_ratio": -1.2585831671001737,
    "exit_led_headline": true
  },
  "generated_utc": "2026-08-01T02:34:51Z"
}```

## runs/r1_qwen_1p5b/analysis/closed_loop_audit.json
```json
{
  "families": {
    "079322feb6": {
      "kind": "exit_only",
      "tau_exit": 0.7,
      "patience_k": 2,
      "n": 1000,
      "frac_exited": 0.95,
      "acc_exited": 0.6926315789473684,
      "acc_not_exited": 0.12,
      "tokens_exited": 1231.822105263158,
      "tokens_not_exited": 14772.9,
      "mean_p_at_exit": 0.8737605660212667,
      "reliability": [
        {
          "bin": [
            0.7,
            0.8
          ],
          "n": 230,
          "mean_p": 0.7548842323862988,
          "acc": 0.6695652173913044
        },
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 281,
          "mean_p": 0.8521845158308851,
          "acc": 0.7117437722419929
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 206,
          "mean_p": 0.9254516137456431,
          "acc": 0.6941747572815534
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 233,
          "mean_p": 0.9714261068295,
          "acc": 0.6909871244635193
        }
      ]
    },
    "33044f6502": {
      "kind": "full",
      "tau_exit": 0.9,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.744,
      "acc_exited": 0.6895161290322581,
      "acc_not_exited": 0.01953125,
      "tokens_exited": 1394.72311827957,
      "tokens_not_exited": 16033.54296875,
      "mean_p_at_exit": 0.9430119738142978,
      "reliability": [
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 446,
          "mean_p": 0.9243227765966424,
          "acc": 0.6704035874439462
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 298,
          "mean_p": 0.9709830542138759,
          "acc": 0.7181208053691275
        }
      ]
    },
    "42c3453b67": {
      "kind": "exit_only",
      "tau_exit": 0.9,
      "patience_k": 2,
      "n": 1000,
      "frac_exited": 0.82,
      "acc_exited": 0.6914634146341463,
      "acc_not_exited": 0.2611111111111111,
      "tokens_exited": 1198.2658536585366,
      "tokens_not_exited": 10221.022222222222,
      "mean_p_at_exit": 0.9479151139172112,
      "reliability": [
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 412,
          "mean_p": 0.9267261562127511,
          "acc": 0.7233009708737864
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 408,
          "mean_p": 0.9693118065011268,
          "acc": 0.6593137254901961
        }
      ]
    },
    "499c0ea86f": {
      "kind": "full",
      "tau_exit": 0.8,
      "patience_k": 2,
      "n": 1000,
      "frac_exited": 0.734,
      "acc_exited": 0.6798365122615804,
      "acc_not_exited": 0.011278195488721804,
      "tokens_exited": 1338.4727520435968,
      "tokens_not_exited": 16127.304511278195,
      "mean_p_at_exit": 0.9188752811838561,
      "reliability": [
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 247,
          "mean_p": 0.8544826420695193,
          "acc": 0.7206477732793523
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 212,
          "mean_p": 0.9261872149863333,
          "acc": 0.6933962264150944
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 275,
          "mean_p": 0.9710747426206415,
          "acc": 0.6327272727272727
        }
      ]
    },
    "5733e5f65d": {
      "kind": "exit_only",
      "tau_exit": 0.95,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.783,
      "acc_exited": 0.722860791826309,
      "acc_not_exited": 0.31336405529953915,
      "tokens_exited": 1130.1673052362707,
      "tokens_not_exited": 9069.917050691245,
      "mean_p_at_exit": 0.9679503923479472,
      "reliability": [
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 783,
          "mean_p": 0.9679503923479472,
          "acc": 0.722860791826309
        }
      ]
    },
    "60948c01b8": {
      "kind": "full",
      "tau_exit": 0.7,
      "patience_k": 2,
      "n": 1000,
      "frac_exited": 0.735,
      "acc_exited": 0.6653061224489796,
      "acc_not_exited": 0.011320754716981131,
      "tokens_exited": 1416.7945578231293,
      "tokens_not_exited": 16192.430188679245,
      "mean_p_at_exit": 0.8819494825642125,
      "reliability": [
        {
          "bin": [
            0.7,
            0.8
          ],
          "n": 154,
          "mean_p": 0.7508341871298753,
          "acc": 0.6883116883116883
        },
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 211,
          "mean_p": 0.8532802671617806,
          "acc": 0.6682464454976303
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 165,
          "mean_p": 0.9274391940145782,
          "acc": 0.7333333333333333
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 205,
          "mean_p": 0.973340495039777,
          "acc": 0.5902439024390244
        }
      ]
    },
    "7b873f923e": {
      "kind": "exit_only",
      "tau_exit": 0.95,
      "patience_k": 2,
      "n": 1000,
      "frac_exited": 0.618,
      "acc_exited": 0.6343042071197411,
      "acc_not_exited": 0.4790575916230366,
      "tokens_exited": 1122.0825242718447,
      "tokens_not_exited": 7006.282722513089,
      "mean_p_at_exit": 0.9692492565097932,
      "reliability": [
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 618,
          "mean_p": 0.9692492565097932,
          "acc": 0.6343042071197411
        }
      ]
    },
    "7ea0180ba8": {
      "kind": "exit_only",
      "tau_exit": 0.9,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.897,
      "acc_exited": 0.7023411371237458,
      "acc_not_exited": 0.1553398058252427,
      "tokens_exited": 1192.076923076923,
      "tokens_not_exited": 12242.59223300971,
      "mean_p_at_exit": 0.9411735324025021,
      "reliability": [
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 558,
          "mean_p": 0.9238472048954297,
          "acc": 0.6935483870967742
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 339,
          "mean_p": 0.9696929741398072,
          "acc": 0.7168141592920354
        }
      ]
    },
    "ab2ad2aa91": {
      "kind": "exit_only",
      "tau_exit": 0.8,
      "patience_k": 2,
      "n": 1000,
      "frac_exited": 0.902,
      "acc_exited": 0.6685144124168514,
      "acc_not_exited": 0.12244897959183673,
      "tokens_exited": 1186.6518847006653,
      "tokens_not_exited": 12682.122448979591,
      "mean_p_at_exit": 0.9131956231303331,
      "reliability": [
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 329,
          "mean_p": 0.8512007220900167,
          "acc": 0.6504559270516718
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 277,
          "mean_p": 0.9268665780659617,
          "acc": 0.6895306859205776
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 296,
          "mean_p": 0.9693086904448431,
          "acc": 0.668918918918919
        }
      ]
    },
    "cbb8895c16": {
      "kind": "exit_only",
      "tau_exit": 0.7,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.976,
      "acc_exited": 0.6762295081967213,
      "acc_not_exited": 0.16666666666666666,
      "tokens_exited": 1167.7571721311476,
      "tokens_not_exited": 12958.833333333334,
      "mean_p_at_exit": 0.8284332508557155,
      "reliability": [
        {
          "bin": [
            0.7,
            0.8
          ],
          "n": 428,
          "mean_p": 0.7477245557809544,
          "acc": 0.6425233644859814
        },
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 306,
          "mean_p": 0.8486297229925791,
          "acc": 0.6666666666666666
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 131,
          "mean_p": 0.9247418742143471,
          "acc": 0.7404580152671756
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 111,
          "mean_p": 0.9702960558839746,
          "acc": 0.7567567567567568
        }
      ]
    },
    "d20644a89a": {
      "kind": "full",
      "tau_exit": 0.8,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.742,
      "acc_exited": 0.6873315363881402,
      "acc_not_exited": 0.011627906976744186,
      "tokens_exited": 1363.3180592991914,
      "tokens_not_exited": 16287.860465116279,
      "mean_p_at_exit": 0.8992043176108615,
      "reliability": [
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 362,
          "mean_p": 0.846225065585658,
          "acc": 0.6767955801104972
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 187,
          "mean_p": 0.9265789045369561,
          "acc": 0.7058823529411765
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 193,
          "mean_p": 0.9720511646468405,
          "acc": 0.689119170984456
        }
      ]
    },
    "dac4beda09": {
      "kind": "full",
      "tau_exit": 0.7,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.778,
      "acc_exited": 0.6542416452442159,
      "acc_not_exited": 0.0,
      "tokens_exited": 1365.4164524421594,
      "tokens_not_exited": 16307.036036036036,
      "mean_p_at_exit": 0.8338514282188563,
      "reliability": [
        {
          "bin": [
            0.7,
            0.8
          ],
          "n": 334,
          "mean_p": 0.7456025691803344,
          "acc": 0.6167664670658682
        },
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 222,
          "mean_p": 0.8484753652198894,
          "acc": 0.6936936936936937
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 103,
          "mean_p": 0.9290709316151814,
          "acc": 0.6796116504854369
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 119,
          "mean_p": 0.9718429917047003,
          "acc": 0.6638655462184874
        }
      ]
    },
    "fdec51004e": {
      "kind": "exit_only",
      "tau_exit": 0.8,
      "patience_k": 1,
      "n": 1000,
      "frac_exited": 0.95,
      "acc_exited": 0.6652631578947369,
      "acc_not_exited": 0.06,
      "tokens_exited": 1232.2221052631578,
      "tokens_not_exited": 13795.2,
      "mean_p_at_exit": 0.8864133218087648,
      "reliability": [
        {
          "bin": [
            0.8,
            0.9
          ],
          "n": 566,
          "mean_p": 0.8452796988689436,
          "acc": 0.6289752650176679
        },
        {
          "bin": [
            0.9,
            0.95
          ],
          "n": 194,
          "mean_p": 0.9247405691859648,
          "acc": 0.6907216494845361
        },
        {
          "bin": [
            0.95,
            1.0
          ],
          "n": 190,
          "mean_p": 0.9698140828233016,
          "acc": 0.7473684210526316
        }
      ]
    }
  },
  "generated_utc": "2026-07-30T17:19:36Z"
}```
