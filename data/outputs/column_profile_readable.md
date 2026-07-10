# TAAC 数据集列级画像报告

- 生成时间: 2026-04-23 11:34:37
- 数据行数: 1000
- 列数: 120
- 标量列: 53
- 列表列: 67

## 类别概览

| category | columns | avg_missing_ratio | scalar_columns | list_columns |
| --- | --- | --- | --- | --- |
| user_int | 46 | 0.2827391304347826 | 35 | 11 |
| domain_b_seq | 14 | 0.012 | 0 | 14 |
| item_int | 14 | 0.21107142857142858 | 13 | 1 |
| domain_c_seq | 12 | 0.002 | 0 | 12 |
| user_dense | 10 | 0.0989 | 0 | 10 |
| domain_d_seq | 10 | 0.08 | 0 | 10 |
| domain_a_seq | 9 | 0.005 | 0 | 9 |
| id | 2 | 0.0 | 2 | 0 |
| time | 2 | 0.0 | 2 | 0 |
| label | 1 | 0.0 | 1 | 0 |

## 类别: domain_a_seq

- 列数: 9
- 平均缺失率: 0.005000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| domain_a_seq_38 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(764162), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_a_seq_39 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(1772695320), np.int64(1772626560), np.int64(1772603941), np.int64(1772553447), np.int64(1772541780), np.int64(1772461020), np.int64(1772459460), np.int64(1772233964)] |
| domain_a_seq_40 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(2), np.int64(2), np.int64(7), np.int64(7), np.int64(2), np.int64(2), np.int64(2), np.int64(12)] |
| domain_a_seq_41 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(6), np.int64(6), np.int64(11), np.int64(11), np.int64(6), np.int64(6), np.int64(6), np.int64(4)] |
| domain_a_seq_42 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(840), np.int64(840), np.int64(0), np.int64(143), np.int64(215), np.int64(840), np.int64(797), np.int64(778)] |
| domain_a_seq_43 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(1365), np.int64(2046), np.int64(0), np.int64(1405), np.int64(583), np.int64(2046), np.int64(1898), np.int64(0)] |
| domain_a_seq_44 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(3792), np.int64(5577), np.int64(0), np.int64(11037), np.int64(0), np.int64(5577), np.int64(0), np.int64(0)] |
| domain_a_seq_45 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(3275), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_a_seq_46 | domain_a_seq 域行为序列特征（list<int64>） | object | list | 0.005 |  |  |  |  |  | 701.086 | 1673.15 | 1888.0 | [np.int64(12), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(14), np.int64(11)] |

## 类别: domain_b_seq

- 列数: 14
- 平均缺失率: 0.012000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| domain_b_seq_67 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(1772720154), np.int64(1772705768), np.int64(1772695970), np.int64(1772695956), np.int64(1772689874), np.int64(1772688611), np.int64(1772687891), np.int64(1772687287)] |
| domain_b_seq_68 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(6), np.int64(22), np.int64(6), np.int64(6), np.int64(6), np.int64(22), np.int64(22), np.int64(22)] |
| domain_b_seq_69 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(58747653), np.int64(139495569), np.int64(139495569), np.int64(143086800), np.int64(58747653), np.int64(58747653), np.int64(58747653)] |
| domain_b_seq_70 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(237), np.int64(471), np.int64(471), np.int64(96), np.int64(237), np.int64(237), np.int64(237)] |
| domain_b_seq_71 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(1724), np.int64(0), np.int64(0), np.int64(0), np.int64(1724), np.int64(1724), np.int64(1724)] |
| domain_b_seq_72 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(7339), np.int64(0), np.int64(0), np.int64(0), np.int64(7339), np.int64(7339), np.int64(7339)] |
| domain_b_seq_73 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_74 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_75 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(16), np.int64(16), np.int64(6), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_76 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_77 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(6), np.int64(6), np.int64(9), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_78 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(1629), np.int64(1629), np.int64(1396), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_79 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(10625), np.int64(10625), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_b_seq_88 | domain_b_seq 域行为序列特征（list<int64>） | object | list | 0.012 |  |  |  |  |  | 570.758 | 1563.15 | 1952.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |

## 类别: domain_c_seq

- 列数: 12
- 平均缺失率: 0.002000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| domain_c_seq_27 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(1772604104), np.int64(1772320822), np.int64(1772234064), np.int64(1771001044), np.int64(1770995005), np.int64(1770791113), np.int64(1770791079), np.int64(1770644055)] |
| domain_c_seq_28 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(31), np.int64(31), np.int64(31), np.int64(31), np.int64(31), np.int64(48), np.int64(31), np.int64(10)] |
| domain_c_seq_29 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(4981970), np.int64(287196), np.int64(255139), np.int64(425092), np.int64(1819562), np.int64(5851938), np.int64(5851938), np.int64(4085610)] |
| domain_c_seq_30 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(316), np.int64(810), np.int64(693), np.int64(693), np.int64(318), np.int64(643), np.int64(643), np.int64(71)] |
| domain_c_seq_31 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(2075)] |
| domain_c_seq_32 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(2)] |
| domain_c_seq_33 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1)] |
| domain_c_seq_34 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(16197), np.int64(318279), np.int64(747893), np.int64(789853), np.int64(47432), np.int64(789853), np.int64(789853), np.int64(789853)] |
| domain_c_seq_35 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(1706), np.int64(1006), np.int64(2734), np.int64(2734), np.int64(128), np.int64(1932), np.int64(1932), np.int64(97)] |
| domain_c_seq_36 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(767838), np.int64(0), np.int64(0), np.int64(0), np.int64(604919), np.int64(0), np.int64(0), np.int64(102119)] |
| domain_c_seq_37 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(6252), np.int64(0), np.int64(2928), np.int64(6747), np.int64(3578), np.int64(0), np.int64(0), np.int64(8612)] |
| domain_c_seq_47 | domain_c_seq 域行为序列特征（list<int64>） | object | list | 0.002 |  |  |  |  |  | 449.431 | 1214.25 | 3894.0 | [np.int64(225864377), np.int64(80211853), np.int64(100281065), np.int64(20753754), np.int64(106973665), np.int64(71727095), np.int64(71727095), np.int64(98860153)] |

## 类别: domain_d_seq

- 列数: 10
- 平均缺失率: 0.080000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| domain_d_seq_17 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3)] |
| domain_d_seq_18 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(51), np.int64(193), np.int64(310), np.int64(310), np.int64(114), np.int64(227), np.int64(18), np.int64(831)] |
| domain_d_seq_19 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(0), np.int64(0), np.int64(1258), np.int64(1258), np.int64(143), np.int64(1599), np.int64(1475), np.int64(951)] |
| domain_d_seq_20 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(0), np.int64(0), np.int64(188), np.int64(188), np.int64(1807), np.int64(0), np.int64(5762), np.int64(0)] |
| domain_d_seq_21 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_d_seq_22 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)] |
| domain_d_seq_23 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(217882), np.int64(290672), np.int64(117875), np.int64(117875), np.int64(90767), np.int64(0), np.int64(469016), np.int64(283767)] |
| domain_d_seq_24 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(0), np.int64(0), np.int64(67), np.int64(0), np.int64(0), np.int64(548), np.int64(511), np.int64(231)] |
| domain_d_seq_25 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(2), np.int64(2), np.int64(2), np.int64(14), np.int64(2), np.int64(7), np.int64(12), np.int64(2)] |
| domain_d_seq_26 | domain_d_seq 域行为序列特征（list<int64>） | object | list | 0.08 |  |  |  |  |  | 1099.859 | 2451.1 | 3951.0 | [np.int64(1772723880), np.int64(1772721120), np.int64(1772721120), np.int64(1772720940), np.int64(1772719800), np.int64(1772719740), np.int64(1772719260), np.int64(1772719200)] |

## 类别: id

- 列数: 2
- 平均缺失率: 0.000000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_id | 物料/内容唯一标识 | int64 | scalar | 0.0 | 837.0 | 6854.0 | 278202253.0 | 112417687.394 | 78049933.907203 |  |  |  | 103989760 |
| user_id | 用户唯一标识 | int64 | scalar | 0.0 | 1000.0 | 2727076.0 | 12728427.0 | 7835799.336 | 2878292.02418 |  |  |  | 3864676 |

## 类别: item_int

- 列数: 14
- 平均缺失率: 0.211071

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_int_feats_10 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 110.0 | 2.0 | 309.0 | 150.007014 | 89.717174 |  |  |  | 64.0 |
| item_int_feats_11 | 物料侧离散/计数型特征（int 或 list<int>） | object | list | 0.439 |  |  |  |  |  | 2.086 | 9.0 | 20.0 | [np.int64(2980), np.int64(2770)] |
| item_int_feats_12 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 352.0 | 0.0 | 2777.0 | 1039.380762 | 779.538065 |  |  |  | 0.0 |
| item_int_feats_13 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 8.0 | 1.0 | 8.0 | 4.456914 | 2.506834 |  |  |  | 6.0 |
| item_int_feats_16 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 662.0 | 2.0 | 35259.0 | 12356.101202 | 8033.661241 |  |  |  | 18629.0 |
| item_int_feats_5 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 82.0 | 4.0 | 325.0 | 118.451904 | 78.534493 |  |  |  | 161.0 |
| item_int_feats_6 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 216.0 | 0.0 | 977.0 | 419.073146 | 245.4726 |  |  |  | 893.0 |
| item_int_feats_7 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 349.0 | 0.0 | 2806.0 | 1052.865731 | 744.977111 |  |  |  | 0.0 |
| item_int_feats_8 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 226.0 | -1.0 | 2431.0 | 463.712425 | 667.147659 |  |  |  | 0.0 |
| item_int_feats_81 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 3.0 | 0.0 | 2.0 | 0.508016 | 0.765904 |  |  |  | 0.0 |
| item_int_feats_83 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.832 | 22.0 | 1.0 | 31.0 | 17.595238 | 9.171149 |  |  |  | 29.0 |
| item_int_feats_84 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.832 | 66.0 | 3.0 | 226.0 | 131.130952 | 67.5317 |  |  |  | 134.0 |
| item_int_feats_85 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.832 | 103.0 | 4.0 | 1001.0 | 439.815476 | 302.799805 |  |  |  | 873.0 |
| item_int_feats_9 | 物料侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 24.0 | 3.0 | 37.0 | 21.171343 | 9.568218 |  |  |  | 30.0 |

## 类别: label

- 列数: 1
- 平均缺失率: 0.000000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_type | 监督标签（当前任务二分类标签） | int32 | scalar | 0.0 | 2.0 | 1.0 | 2.0 | 1.124 | 0.329582 |  |  |  | 1 |

## 类别: time

- 列数: 2
- 平均缺失率: 0.000000

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_time | 标签事件时间（用于时序切分） | int64 | scalar | 0.0 | 553.0 | 1772725027.0 | 1772725910.0 | 1772725503.904 | 196.391153 |  |  |  | 1772725413 |
| timestamp | 样本记录时间戳 | int64 | scalar | 0.0 | 501.0 | 1772725000.0 | 1772725781.0 | 1772725275.446 | 193.778144 |  |  |  | 1772725140 |

## 类别: user_dense

- 列数: 10
- 平均缺失率: 0.098900

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| user_dense_feats_61 | 用户侧稠密向量特征（list<float>） | object | list | 0.002 |  |  |  |  |  | 255.488 | 256.0 | 256.0 | [np.float32(0.10862021), np.float32(-0.020586327), np.float32(0.09290086), np.float32(0.029013816), np.float32(-0.036376655), np.float32(0.040550303), np.float32(0.047041714), np.float32(0.040784705)] |
| user_dense_feats_62 | 用户侧稠密向量特征（list<float>） | object | list | 0.07 |  |  |  |  |  | 2.114 | 4.0 | 5.0 | [np.float32(31722.5), np.float32(6254.6665)] |
| user_dense_feats_63 | 用户侧稠密向量特征（list<float>） | object | list | 0.07 |  |  |  |  |  | 2.555 | 6.0 | 11.0 | [np.float32(60427.0), np.float32(17337.0), np.float32(3018.0), np.float32(713.5)] |
| user_dense_feats_64 | 用户侧稠密向量特征（list<float>） | object | list | 0.07 |  |  |  |  |  | 3.822 | 9.0 | 18.0 | [np.float32(31722.5), np.float32(17337.0), np.float32(713.5)] |
| user_dense_feats_65 | 用户侧稠密向量特征（list<float>） | object | list | 0.08 |  |  |  |  |  | 5.713 | 17.0 | 49.0 | [np.float32(17337.0), np.float32(3018.0), np.float32(1129.0), np.float32(298.0)] |
| user_dense_feats_66 | 用户侧稠密向量特征（list<float>） | object | list | 0.086 |  |  |  |  |  | 7.139 | 21.05 | 66.0 | [np.float32(17337.0), np.float32(3018.0), np.float32(1129.0), np.float32(298.0)] |
| user_dense_feats_87 | 用户侧稠密向量特征（list<float>） | object | list | 0.015 |  |  |  |  |  | 315.2 | 320.0 | 320.0 | [np.float32(-0.1093), np.float32(-0.0052), np.float32(0.0193), np.float32(-0.197), np.float32(0.0865), np.float32(0.0533), np.float32(0.2245), np.float32(0.0311)] |
| user_dense_feats_89 | 用户侧稠密向量特征（list<float>） | object | list | 0.055 |  |  |  |  |  | 9.45 | 10.0 | 10.0 | [np.float32(-0.0044), np.float32(-0.1074), np.float32(0.1108), np.float32(0.7048), np.float32(0.2637), np.float32(-0.5479), np.float32(-0.3195), np.float32(-0.0867)] |
| user_dense_feats_90 | 用户侧稠密向量特征（list<float>） | object | list | 0.091 |  |  |  |  |  | 9.09 | 10.0 | 10.0 | [np.float32(-0.0042), np.float32(-0.096), np.float32(0.0853), np.float32(0.6986), np.float32(0.2819), np.float32(-0.5583), np.float32(-0.312), np.float32(-0.0827)] |
| user_dense_feats_91 | 用户侧稠密向量特征（list<float>） | object | list | 0.45 |  |  |  |  |  | 5.5 | 10.0 | 10.0 | [np.float32(-0.0065), np.float32(-0.2164), np.float32(0.3663), np.float32(0.7118), np.float32(0.0537), np.float32(-0.396), np.float32(-0.3708), np.float32(-0.1218)] |

## 类别: user_int

- 列数: 46
- 平均缺失率: 0.282739

| column | description | dtype | value_mode | missing_ratio | unique_count | min | max | mean | std | len_mean | len_p95 | len_max | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| user_int_feats_1 | 用户侧离散/计数型特征（int 或 list<int>） | int64 | scalar | 0.0 | 3.0 | 1.0 | 4.0 | 3.381 | 1.169546 |  |  |  | 4 |
| user_int_feats_100 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.845 | 2.0 | 1.0 | 2.0 | 1.954839 | 0.207658 |  |  |  | 2.0 |
| user_int_feats_101 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.91 | 2.0 | 2.0 | 3.0 | 2.955556 | 0.20608 |  |  |  | 3.0 |
| user_int_feats_102 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.877 | 2.0 | 1.0 | 3.0 | 1.130081 | 0.493195 |  |  |  | 1.0 |
| user_int_feats_103 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.862 | 3.0 | 1.0 | 3.0 | 2.717391 | 0.551534 |  |  |  | 3.0 |
| user_int_feats_104 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.372 | 3.0 | 1.0 | 3.0 | 2.359873 | 0.929582 |  |  |  | 3.0 |
| user_int_feats_105 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.309 | 3.0 | 1.0 | 3.0 | 2.286541 | 0.470958 |  |  |  | 2.0 |
| user_int_feats_106 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.16 | 3.0 | 1.0 | 3.0 | 1.759524 | 0.430149 |  |  |  | 2.0 |
| user_int_feats_107 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.3 | 2.0 | 1.0 | 2.0 | 1.094286 | 0.292226 |  |  |  | 1.0 |
| user_int_feats_108 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.516 | 6.0 | 2.0 | 7.0 | 5.454545 | 1.687337 |  |  |  | 6.0 |
| user_int_feats_109 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.854 | 7.0 | 1.0 | 7.0 | 2.993151 | 2.359756 |  |  |  | 1.0 |
| user_int_feats_15 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.139 |  |  |  |  |  | 3.232 | 8.0 | 13.0 | [np.int64(928), np.int64(556), np.int64(538), np.int64(739), np.int64(94)] |
| user_int_feats_3 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.03 | 341.0 | 9.0 | 1839.0 | 987.556701 | 516.784036 |  |  |  | 1753.0 |
| user_int_feats_4 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.03 | 268.0 | 1.0 | 986.0 | 498.813402 | 299.241927 |  |  |  | 6.0 |
| user_int_feats_48 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.002 | 52.0 | 3.0 | 99.0 | 58.006012 | 27.543152 |  |  |  | 42.0 |
| user_int_feats_49 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.007 | 2.0 | 1.0 | 2.0 | 1.582075 | 0.493218 |  |  |  | 2.0 |
| user_int_feats_50 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.004 | 2.0 | 0.0 | 1.0 | 0.997992 | 0.044766 |  |  |  | 1.0 |
| user_int_feats_51 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.001 | 5.0 | 40.0 | 150.0 | 56.157157 | 3.579545 |  |  |  | 56.0 |
| user_int_feats_52 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.001 | 36.0 | 5.0 | 174.0 | 93.855856 | 56.313182 |  |  |  | 24.0 |
| user_int_feats_53 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.001 | 264.0 | 3.0 | 557.0 | 288.541542 | 165.284443 |  |  |  | 101.0 |
| user_int_feats_54 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.368 | 462.0 | 3.0 | 2843.0 | 1476.783228 | 838.031814 |  |  |  | 810.0 |
| user_int_feats_55 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.019 | 13.0 | 8.0 | 41.0 | 29.681957 | 9.644379 |  |  |  | 41.0 |
| user_int_feats_56 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.019 | 405.0 | 1.0 | 1434.0 | 752.657492 | 405.371813 |  |  |  | 681.0 |
| user_int_feats_57 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.031 | 105.0 | 2.0 | 250.0 | 126.588235 | 67.580191 |  |  |  | 111.0 |
| user_int_feats_58 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.15 | 2.0 | 1.0 | 2.0 | 1.698824 | 0.458769 |  |  |  | 1.0 |
| user_int_feats_59 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.15 | 8.0 | 1.0 | 14.0 | 8.370588 | 4.54782 |  |  |  | 3.0 |
| user_int_feats_60 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.592 |  |  |  |  |  | 0.617 | 2.0 | 2.0 | [np.int64(2)] |
| user_int_feats_62 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.07 |  |  |  |  |  | 2.114 | 4.0 | 5.0 | [np.int64(6), np.int64(4)] |
| user_int_feats_63 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.07 |  |  |  |  |  | 2.555 | 6.0 | 11.0 | [np.int64(9), np.int64(15), np.int64(45), np.int64(36)] |
| user_int_feats_64 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.07 |  |  |  |  |  | 3.822 | 9.0 | 18.0 | [np.int64(2), np.int64(50), np.int64(23)] |
| user_int_feats_65 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.08 |  |  |  |  |  | 5.713 | 17.0 | 49.0 | [np.int64(46), np.int64(94), np.int64(166), np.int64(183)] |
| user_int_feats_66 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.086 |  |  |  |  |  | 7.139 | 21.05 | 66.0 | [np.int64(649), np.int64(63), np.int64(1383), np.int64(18)] |
| user_int_feats_80 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.2 |  |  |  |  |  | 1.158 | 3.0 | 5.0 | [np.int64(8), np.int64(6)] |
| user_int_feats_82 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.204 | 23.0 | 1.0 | 23.0 | 9.096734 | 6.560077 |  |  |  | 18.0 |
| user_int_feats_86 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.692 | 61.0 | 2.0 | 245.0 | 105.474026 | 74.176754 |  |  |  | 37.0 |
| user_int_feats_89 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.055 |  |  |  |  |  | 9.45 | 10.0 | 10.0 | [np.int64(0), np.int64(9), np.int64(9), np.int64(2), np.int64(2), np.int64(8), np.int64(2), np.int64(3)] |
| user_int_feats_90 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.091 |  |  |  |  |  | 9.09 | 10.0 | 10.0 | [np.int64(0), np.int64(9), np.int64(2), np.int64(5), np.int64(4), np.int64(3), np.int64(3), np.int64(2)] |
| user_int_feats_91 | 用户侧离散/计数型特征（int 或 list<int>） | object | list | 0.45 |  |  |  |  |  | 5.5 | 10.0 | 10.0 | [np.int64(0), np.int64(4), np.int64(9), np.int64(3), np.int64(5), np.int64(4), np.int64(6), np.int64(9)] |
| user_int_feats_92 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.494 | 2.0 | 1.0 | 2.0 | 1.5 | 0.5 |  |  |  | 1.0 |
| user_int_feats_93 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.171 | 36.0 | 1.0 | 37.0 | 14.667069 | 8.46237 |  |  |  | 3.0 |
| user_int_feats_94 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.521 | 6.0 | 1.0 | 6.0 | 3.770355 | 2.101167 |  |  |  | 1.0 |
| user_int_feats_95 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.318 | 3.0 | 1.0 | 3.0 | 2.758065 | 0.438407 |  |  |  | 3.0 |
| user_int_feats_96 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.678 | 3.0 | 1.0 | 3.0 | 1.81677 | 0.40259 |  |  |  | 1.0 |
| user_int_feats_97 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.292 | 3.0 | 1.0 | 3.0 | 1.59887 | 0.881446 |  |  |  | 3.0 |
| user_int_feats_98 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.103 | 3.0 | 1.0 | 3.0 | 2.677815 | 0.587769 |  |  |  | 3.0 |
| user_int_feats_99 | 用户侧离散/计数型特征（int 或 list<int>） | float64 | scalar | 0.812 | 2.0 | 1.0 | 3.0 | 2.93617 | 0.351547 |  |  |  | 3.0 |
