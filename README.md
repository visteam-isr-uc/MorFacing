# MorFacing: Face Morphing Robustness benchmarking

This repo includes functionality for morphing attack detection benchmarking from the paper "MorDeephy: Face Morphing Detection Via Fused Classification".
The project may be used for performing benchmarks on developer side and further submitting results for comparison between different developers.
The project only unifies the data, protocols, performance estimation, results comparison. 
**The extracting of the predictions data is up to the developer.**
We only propose an example for generating random predictions [generate predictions of test_model](./mf_utils/generate_predictions_test_model.py).

## Executing Benchmark

1. [Get](#Data) and align them to match your model.
(Or if suitable for your model use [aligned data](#Data)). 
To align the data we use [MTCNN](https://github.com/ipazc/mtcnn).
We propose to have separate directories for different types of alignment.

2. Implement the function for extracting model prediction from image path in extract_prediction.py. 
The name of the function must match the name of the model.

3. Execute the benchmark for your model.
```
python run_benchmark.py --models_path "./models/" --model_name "name_of_your_model" 

```

## Results comparison

To compare several results and plot together curves for defined protocols, run 
the script specifying protocol label and models for comparison:
```
python compare_models_mmpmr.py  
python compare_models_mmpmr.py -m ArcFace_R50_ms1mv2 MagFace_R50_ms1mv2 -d "ArcFace" "MagFace" -l stylegan -r MMPMR -f FMR

python compare_models_rmmr.py  
python compare_models_rmmr.py -m ArcFace_R50_ms1mv2 MagFace_R50_ms1mv2 -d "ArcFace" "MagFace" -l stylegan -r MMPMR

```

Result curves will appear in the ```/combined_results```.


## Submitting
### How to submit your results

Make a pull request with your model folder (only with the results) to this repo
  
 ```
 Detailed manual Soon
 ```

## Generaing new morphs

To extend this toolset with  (see [Contacts](#Contacts) )

1. [Get data](#Data).

2. Generate morphs according to the [protocol](./morphing_pairing_protocols/morphing_protocol_N5.txt)

### Sharing new morphs
If you want to contribute and extend this toolset with your morphs (generated with your method):

Send them to [Contact email](#Contacts and versioning) with the shared link from the same email domain that was used for the data request.


## Data

To get the actual version of the data please sign a licence agreement and send it to the [Contact email](#Contacts and versioning)


## Licence agreement
 ```
 Soon
 ```

If use of our work in your research, please cite the paper in your publications:
```
@INPROCEEDINGS{10744449,
  author={Medvedev, Iurii and Gonçalves, Nuno},
  booktitle={2024 IEEE International Joint Conference on Biometrics (IJCB)}, 
  title={MorFacing: A Benchmark for Estimation Face Recognition Robustness to Face Morphing Attacks}, 
  year={2024},
  volume={},
  number={},
  pages={1-10},
  keywords={Measurement;Printing;Deep learning;Protocols;Face recognition;Estimation;Authentication;Benchmark testing;Robustness;Security;face morphing (FM);face recognition (FR);face recognition system(FRS) computer vision;deep learning},
  doi={10.1109/IJCB62174.2024.10744449}}

```

## Contacts and versioning

| Contact email   |  iurii.medvedev@isr.uc.pt   |
| Version         |   1.0   | 




## Acknowledgements
The authors would like to thank the Portuguese Mint and Official Printing Office (INCM) and the 
[Institute of Systems and Robotics - University of Coimbra](https://www.isr.uc.pt) for the support of the project Facing. 
[This work has been supported by Fundação para a Ciência e a Tecnologia (FCT)](https://www.fct.pt/) under the 
project UIDB/00048/2020 and 2022.11941.BD.  The computational part of this work was performed with the support of 
NVIDIA Applied Research Accelerator Program with hardware and software provided by [NVIDIA](https://developer.nvidia.com/higher-education-and-research).



<p float="left">
  <img src="./logos/ISR_logo.png" alt="ISR" width="15%";"/>
  <img src="./logos/UC_logo.jpg" alt="UC" width="30%";"/>
  <img src="./logos/FCT_Logo.jpg" alt="FCT" width="20%";"/>
  <img src="./logos/nvidia_logo.png" alt="NVIDIA" width="32%";"/>
</p>

