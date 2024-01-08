Some of these models have been uploaded to GitHub, and some not, depending among others
on size constraints.

`baseline_4_IX.pt` - baseline model (class BaselineBertDetector), model prajjwal1/bert-tiny, 
max_length 512, title, author, text, trained 4 epochs, lr 1e-4, eval F1: 0.998

`baseline_6_IX.pt` - as above, but author excluded

`welfake_32.pt` - initial of a bert-base-uncased model on the WELFake dataset, max_length 512,
trained 32 epochs but converged much quicker, lr: 1e-4, eval F1: 0.67 class WelfakeDetector

`welfake_1e-5.pt` - as above but trained for 4 epochs with a lr of, surprise surprise, 1e-5. F1 of 0.996
again.
