# Explore use cases with MultiModalility Model on SageMaker.

[Visual Dialog(VisDial)](https://visualdialog.org/) requires an AI agent to hold a meaningful dialog with humans in natural, conversational language about visual content. Specifically, given an image, a dialog history, and a follow-up question about the image, the task is to answer the question.

[Vision-languagepre-training(VLP)](https://arxiv.org/pdf/2210.09263.pdf) 

[BLIP2](https://arxiv.org/pdf/2201.12086.pdf) is a unified VLP framework

## Content Moderation
We use dataset from [Content Moderation with AWS AI Services](https://github.com/aws-samples/amazon-rekognition-code-samples/tree/main/content-moderation) to test how BLIP2 can detect unsafe content in the image and meanwhile give the explanation with effective prompts.

