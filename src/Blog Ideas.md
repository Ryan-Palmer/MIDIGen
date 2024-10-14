# Intro

In this blog we will explore modern machine learning tools and techniques using music generation as a motivation, rather than the more common text based scenario.

Although we will use the same tools and techniques, building a 'GPT for music' adds a few extra challenges which will keep us on our toes and require thinking outside the box more than a little (hint - it has multiple layers and a time dimension!).

We will look at topics such as

- Translation of raw data into a suitable format for training
- Efficient encoding / decoding to allow processing of larger data sets
- Batching, to segment data for the model during training
- Attention-based Transformer models which have contextual understanding

I am going to try to keep it quite high level, but for those interested in taking a look under the covers and digging through code you can [grab the workbooks from my Github](https://github.com/Ryan-Palmer/MIDIGen).

## Motivation / History
If you have followed my [previous blogs](https://www.compositional-it.com/news-blog/author/ryan/) you may have noticed that I have a keen interest in all things machine learning. 

This fascination began in 2013 when I wrote my [university dissertation](https://1drv.ms/b/c/91fc7a2609794446/EUZEeQkmevwggJEZBQAAAAABtwH2qjO3hw2U96w6LA3Ytw?e=p5ilV1) on the topic of generative music. This culminated in a [prototype instrument](https://www.youtube.com/watch?v=J-LFz0P3Uto&t=89s&ab_channel=RyanPalmer), coded first in MaxMSP and then Python, which captured statistics about a performance and then generated more music in the same style.

At the time, machine learning and artificial intelligence were terms more often discussed in academia or even sci-fi rather than business or software circles. I was really just a novice programmer naively exploring a thread of ideas from first principles, not really aware of or building upon any previous work. Fast forward 10 years and the world looks rather different. We are witnessing an explosion of technology and ideas which are at once exciting and fascinating is the possibilities they unlock, and also often overwhelming or worrying for the change they will bring.

I am a believer that knowledge is power - if you understand things, you can have a say in them. We all share the responsibility to both educate ourselves and help others climb that ladder, as this gives them more of a voice and will ultimately benefit us all. Most things can be understood by most people if can find a way to see the world from their perspective.

With that in mind I have been racing to catch up and keep up with all of the developments in the ML space which is no mean feat, given the pace of the industry. Having done a lot of cramming with great books such as [Hands on Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) and excellent video resources such as [Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) and [Statquest](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)'s channels, I needed a personal project to really embed the knowledge in my mind. What better than picking up where I left off 11 years ago with music generation, but using all of the modern tools and techniques?

And that brings us to this blog! I hope to at the very least highlight all the wonderful people and resources which helped me along the way. Additionally, I would love to help to de-mystify the topic and show that it really is accessible and understandable to anyone with a curious mind, both technical and non-technical.

In addition to the resources linked above, this project leant heavily on Andrew Shaw's [MusicAutoBot](https://github.com/bearpelican/musicautobot/tree/master) project and Nick Ryan's [Coding a Paper](https://www.youtube.com/playlist?list=PLam9sigHPGwOe8VDoS_6VT4jjlgs9Uepb) series, which themselves were built upon the shoulders of giants. Thanks guys!


# Basics

## MIDI format

## Tools to load and visualise

## Encoding / Decoding

## Cross Entropy loss


# Transformers

## Attention

## Embeddings

## Evaluation


# Transformer Memory

## Recurrent (XL) memory

## RAG (KNN) memory


# Batches

## One long seq vs per-song vs contiguous

## Data location (In memory vs on disk and GPU vs CPU)

## Multi-threaded encoding


# Byte Pair / Action Encoding

## Motivation

## Challenges (time dimension)

## Findings


# Conclusion

## Learning

## Music Generation

## Next Steps