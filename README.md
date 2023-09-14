# Lumos
This repository releases a dataset about the throughput and delivery time of adaptive video streaming, which was collected in real-world mobile networks from December 2019 to May 2021 and used in the following papers:

- Gerui Lv, Qinghua Wu, Weiran Wang, Zhenyu Li, and Gaogang Xie, "[Lumos: towards Better Video Streaming QoE through Accurate Throughput Prediction](https://ieeexplore.ieee.org/abstract/document/9796948)," _IEEE INFOCOM 2022._

- Gerui Lv, Qinghua Wu, Qingyue Tan, Weiran Wang, Zhenyu Li, and Gaogang Xie, "[Accurate Throughput Prediction for Improving QoE in Mobile Adaptive Streaming](https://ieeexplore.ieee.org/abstract/document/10246426)," _IEEE Transactions on Mobile Computing (TMC) 2023._

## Dataset Statistics
The dataset contains 590 video sessions running ABR algorithms (`ABR_data`, 392 sessions) and with constant bitrate level (`constant_data`, 198 sessions). Each session typically has a 5-minute or 5.5-minute playback duration. The following table provides detailed statistics. For more information, please refer to our paper.

| **Type** | **Downstream Bandwidth** | **Connection Type** | **Signal Strength** |
| --- | --- | --- | --- |
| ABR data (392) | 50Mbps: 198<br>5Mbps: 194 | Wi-Fi: 237<br>4G: 155 | Strong: 141<br>Medium: 114<br>Weak: 137 |
| Constant data (198) | 50Mbps: 120<br>5Mbps: 78 | Wi-Fi: 144<br>4G: 54 | Strong: 162<br>Medium: 6<br>Weak: 30 |

## Dataset Format
### ABR Data
#### File name format
`[Test time]-[Downstream bandwidth]-[Connection type]-[Signal strength]-[Video]-[ABR algorithm].csv`

- `Test time`: start time of a test; the format in Python strftime() and strptime() is "%y%m%d_%H%M"
- `Downstream bandwidth`: ["5Mbps", "50Mbps"], the limited bandwidth of the server
- `Connection type`: ["wifi_2.4GHz", "wifi_5GHz", "4g"]
- `Signal strength`: ["strong", "medium", "weak"]
- `Video`: the format is [Video name]_[Chunk duration]; Video name is in ["bbb", "ed"] (Big Buck Bunny and Elephant Dream), and Chunk duration is in ["2s", "4s"]
- `ABR algorithm`: including RB, BBA (SIGCOMM '14), MPC and RobustMPC (SIGCOMM '15), Pensieve (SIGCOMM '17, retrained with our dataset), and HYB (described in Oboe, SIGCOMM '18)

Examples:

- 210415_1317-50Mbps-wifi_2.4GHz-medium-ed_4s-Pensieve.csv
- 210422_0115-5Mbps-wifi_5GHz-strong-bbb_4s-MPC.csv
- 210426_1225-50Mbps-4g-weak-bbb_4s-BBA.csv
#### Data format
Each .csv file corresponds to the data of a video session. The bitrate of each chunk is determined by the ABR algorithm. For each row from the 2nd line (the 1st line is the title filed) in the file, the format is as follows:

`[downstream_bandwidth],[connection_type],[signal_strength],[bitrate(Kbps)],[chunk_size(KBytes)],[app_throughput(Kbps)],[delivery_time(s)],[player_state],[relative_chunk_index]`

Examples:

- 5M,wifi,weak,4300,20137,6965,2.891,steady,1
- 50M,4g,strong,300,1223,7598,0.161,buffering,5
### Constant Data
#### File name format
`[Test time]-[Downstream bandwidth]-[Connection type]-[Signal strength]-[Video]-[Bitrate].csv`

- `Test time`, `Downstream bandwidth`, `Signal strength`, and `Video` are the same as in ABR data
- `Connection type`: ["wifi", "4g"]
- `Bitrate`: ["300Kbps", "750Kbps", "1200Kbps", "1850Kbps", "2850Kbps", "4300Kbps"]

Examples:

- 200426_1330-50Mbps-wifi-strong-bbb_4s-300Kbps.csv
- 200502_1107-50Mbps-4g-medium-bbb_4s-4300Kbps.csv
- 200504_1000-5Mbps-wifi-weak-bbb_4s-1200Kbps.csv
#### Data format
Each .csv file corresponds to the data of a video session at a constant bitrate. The data format is the same as in ABR data except for `relative_chunk_index`. In constant data, `relative_chunk_index` stays 0 for the steady state while increasing from 1 for the buffering state.

Examples:

- 50M,4g,weak,2850,15,116,0.125,steady,0
- 50M,wifi,strong,1200,6377,15746,0.405,buffering,1


