# AudioLM


# update data cache


gsutil -m cp -R google/src/cloud/tianshenghuang/tiansheng2/google3/cache  gs://xcloud-shared/tianshenghuang


from datasets import load_dataset
ds = load_dataset("anonymous4486/audio_beavertail_30k_test", cache_dir="google/src/cloud/tianshenghuang/tiansheng2/google3/cache")


<!-- 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 -->

<!-- mixed training -->
sbatch grpo_safety_mix.sh 0
<!-- sbatch grpo_safety_mix.sh 0.05 -->
sbatch grpo_safety_mix.sh 0.1
sbatch grpo_safety_mix.sh 0.2
sbatch grpo_safety_mix.sh 0.5

<!-- sbatch grpo_safety_mix.sh 0.7 -->
<!-- gsm8k -->
<!-- evaluate safety of base model  -->
sbatch safety_test.sh Qwen/Qwen2-Audio-7B-Instruct

<!-- evaluate safety of reasoning train -->
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1


<!-- evaluate amse of base model  -->
sbatch safety_test_advwave.sh Qwen/Qwen2-Audio-7B-Instruct
<!-- evaluate safety of reasoning train -->
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.1

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.2

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.6

<!-- sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_1 -->

<!-- evaluate accuracy of base model  -->
sbatch gsm8k_test.sh Qwen/Qwen2-Audio-7B-Instruct

<!-- evaluate accuracy of reasoning train -->
<!-- sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.05
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.1
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.2
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_1 -->


sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.1
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.2
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.6


sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
<!-- sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.8 -->
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.99
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1


<!-- gsm8k for noisy trained model -->
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise_noisy_sft_mixture_0.9


sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_1


<!-- sft  -->
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.99
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.8
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.99
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.8

sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.3
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.7

<!-- sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.3
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.7 -->
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.2
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test_advwave.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1

sbatch safety_test_advwave.sh Qwen/Qwen2-Audio-7B-Instruct

sbatch safety_test_advwave_whitebox.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave_whitebox.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.2
sbatch safety_test_advwave_whitebox.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch safety_test_advwave_whitebox.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test_advwave_whitebox.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1



sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.8
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1

sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.99
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1


<!-- sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.2
sbatch safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5 -->






<!-- advwave attack sft-->
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.2
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1
sbatch advwave_attack.sh  Qwen/Qwen2-Audio-7B-Instruct

sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave_jaml.sh  Qwen/Qwen2-Audio-7B-Instruct


<!-- noise optimization  (suffix_length)  -->
sbatch optimize_noise.sh  40

<!-- noisy sft train  -->
sbatch noisy_sft_mix.sh 0 
sbatch noisy_sft_mix.sh 0.2
sbatch noisy_sft_mix.sh 0.5
sbatch noisy_sft_mix.sh 0.9
sbatch noisy_sft_mix.sh 1

<!-- sft train  -->
sbatch sft_safety_mix.sh 0 
sbatch sft_safety_mix.sh 0.2
sbatch sft_safety_mix.sh 0.5
sbatch sft_safety_mix.sh 0.9
sbatch sft_safety_mix.sh 1


<!-- whitebox advwave attack sft  -->
sbatch advwave_attack.sh  Qwen/Qwen2-Audio-7B-Instruct
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0 50000
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1 50000
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5 50000
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9 50000
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1 50000

sbatch advwave_attack.sh  Qwen/Qwen2-Audio-7B-Instruct
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0 
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1 
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5 
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9 
sbatch advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1 




<!-- whitebox advwave attack adversarial training  -->
sbatch advwave_attack.sh  Qwen/Qwen2-Audio-7B-Instruct

sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.9
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.9
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.9
sbatch  advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.9


sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.5 50000
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.5 50000
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5 50000
sbatch  advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.5 50000

sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.5 
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.5 
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5 
sbatch  advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.5 

sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.1
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.1
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.1
sbatch  advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.1

sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_1
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_1
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_1 50000
sbatch  advwave_attack.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_1

<!-- suffix len exp -->
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5 100000
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5 100000

sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5 150000
sbatch  advwave_attack.sh  ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5 150000



<!-- blackbox advwave attack sft -->
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1

sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.9
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.9
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.9
sbatch  safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.9

sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.5
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.5
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5
sbatch  safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.5

sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.1
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.1
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.1
sbatch  safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.1

sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_1
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_1
sbatch  safety_test_advwave_jaml.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_1
sbatch  safety_test_advwave_jaml.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_1


<!-- gsm8k noisy sft  -->

<!-- gsm8k sft  -->
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1



<!-- gsm8k adversarial training  -->
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_1
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_1
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_1
sbatch  gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_1

sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.9
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.9
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.9
sbatch  gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.9

sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.5
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.5
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5
sbatch  gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.5

sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.1
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.1
sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.1
sbatch  gsm8k_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.1


sbatch  gsm8k_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0


sbatch mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
sbatch mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1
sbatch mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5
sbatch mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9
sbatch mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1

sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.5
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.5
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5
sbatch  mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.5

sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.9
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.9
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.9
sbatch  mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.9

sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.1
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.1
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.1
sbatch  mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.1


sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_1
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_1
sbatch  mmlu_biology_test.sh  ../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_1
sbatch  mmlu_biology_test.sh ../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_1


sbatch  safety_test.sh  ../../ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.5
sbatch  safety_test.sh  ../../ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.5
sbatch  safety_test.sh  ../../ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5
sbatch  safety_test.sh ../../ckpt/Qwen2-Audio-7B-Instruct_noise2_100_0.5



sbatch optimize_noise_2.sh  0 0.1
sbatch optimize_noise_2.sh  1 0.1
sbatch optimize_noise_2.sh  10 0.1
sbatch optimize_noise_2.sh  100 0.1

sbatch optimize_noise_2.sh  10 0
sbatch optimize_noise_2.sh  10 1

sbatch optimize_noise_2.sh  0 0.9
sbatch optimize_noise_2.sh  1 0.9
sbatch optimize_noise_2.sh  10 0.9
sbatch optimize_noise_2.sh  100 0.9

