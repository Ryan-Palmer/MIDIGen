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

And that brings us to this blog! I hope to at the very least highlight all the wonderful people and leave a breadcrumb trail to the resources which helped me along the way. Additionally, I would love to help to de-mystify the topic and show that it really is accessible and understandable to anyone with a curious mind, both technical and non-technical.

In addition to the resources linked above, this project leant heavily on Andrew Shaw's [MusicAutoBot](https://github.com/bearpelican/musicautobot/tree/master) project and Nick Ryan's [Coding a Paper](https://www.youtube.com/playlist?list=PLam9sigHPGwOe8VDoS_6VT4jjlgs9Uepb) series, which themselves were built upon the shoulders of giants. Thanks guys!


# Basics

## MIDI format

The [MIDI format](https://en.wikipedia.org/wiki/MIDI) (**M**usical **I**nstrument **D**igital **I**nterface) was developed in the early 1980s as a universal data and hardware standard which allows devices and software developed by different manufacturers to share a common language.

The data and its associated file format(s) effectively represent a digital music score which can be played back on any instrument(s) that support the standard.

Its longevity and popularity make MIDI an ideal source of data for a machine learning project. Also, unlike audio data, the score for a piece of music takes up a relatively tiny space and so much more can be loaded into memory and processed quickly. They mostly comprise of note on / off events and performance information (e.g pitchbend) along with some metadata describing things such as instrument choice and tempo.

There are lots of great sources of MIDI files if you [look around on the internet](https://github.com/albertmeronyo/awesome-midi-sources). I began with a relatively small set of [video game](https://www.vgmusic.com/) music and eventually worked with the entire [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) which comprises around 200,000 songs in almost every style you can imagine.

## Tools to load and visualise

There are some great libraries available for loading and working with MIDI files in Python.

This project mainly uses [Music21](https://www.music21.org/music21docs/about/what.html) which is very mature and fully featured. It allows you to load and save MIDI files, inspecting and changing their contents in its high level 'Stream' format.

It also works well with [MuseScore](https://musescore.org/en) to render a piano-roll timeline or classical notation in the output cells of your [Jupyter Notebook](https://jupyter.org/). I used notebooks throughout this project to interleave code, output and thoughts / documentation (albeit via the [VSCode](https://code.visualstudio.com/) [Polyglot Notebooks extension](https://code.visualstudio.com/docs/languages/polyglot))

Another python library I used is [pretty-midi](https://craffel.github.io/pretty-midi/) which has a great API and works well with [FluidSynth](https://www.fluidsynth.org/)'s synthesis engine to render the scores.

## Encoding / Decoding

The MIDI data can't be fed directly into a machine learning model - at least not the kind we will be looking at. It first needs to be broken up into a series of tokens, which are each assigned a number. For instance, if we assigned `A=1`, `B=2`, `C=3`, `D=4` and `E=5`, then the words `CAB ACE` would be represented as `312 135`.

Sounds easy enough - however, as hinted in the intro, this is where using music data rather than text presents its first challenge.

- More than one note may be playing simultaneously
- Notes have a position in time relative to each other

The encoding process was therefore a bit more involved. It comprised of three steps:

1. Sparse Score

Convert the MIDI file into a giant array which held a value for every pitch at every step in time (and for every instrument!). The value describes if a note was started for how long (so a zero means 'no note' and a 4 means 'start a note which lasts for 4 steps').

It is referred to as 'sparse' as it is quite literally nearly empty, as most steps on most instruments are zero.

For example a single instrument might have

```
4 0 0 0 0 0 0 0 //... up to final timestep of song
0 0 0 0 0 0 2 0
0 2 0 0 0 0 2 0
//... up to 128 rows
```

This takes 24 values to show

- Step 1, Pitch 1 = start of a 4-step note
- Step 2, Pitch 3 = start of a 2-step note
- Step 7, Pitch 2 and 3 = start of a 2-step note.

2. Position Score

The sparse score isn't very memory or processing efficient, as it contains very little information given its size.

We could, alternatively, just record when each note starts and for how long. This creates a much more compressed form of score. The previous sparse example can be re-written using 14 tokens as

```
1, 4
-1, 1
3, 2
-1, 5
2, 2
3, 2
-1, 2
```

where `-1` represents a gap before the next note.

3. Index Score

Now we just need to flatten all these values into a single list of tokens so we can feed it into our model, just like the text example of `CAB ACE` earlier.

There are all sorts of different encoding schemes you could employ, many of which can be seen on the [MIDITok](https://miditok.readthedocs.io/en/latest/tokenizations.html) website (which I only just discovered!). What they more or less all have in common, including the one I used which was adapted from MusicAutoBot, are

- Tokens for each note (of the 128 available in MIDI)
- Tokens for each duration (from a single timestep all the way up to whatever limit you set on note length).
- A few 'special' tokens for the start and end of a song, and for a gap of silence.

Our example can now be rewritten as

```
<sos><n1><d4><sep><d1><n3><d2><sep><d5><n2><d2><n3><d2><sep><d2><eos>
```

These tokens are each assigned a number and that's it, our data is encoded and ready to go.

To decode the data we just follow the reverse of this process, turning tokens into positions and positions into a sparse score before finally converting the sparse score into MIDI.


## Measuring performance

Just like in essence a model like GPT-2 is a 'next word predictor', so we are building a 'next note predictor'. We are going to feed in a sequence of tokens and as the computer to predict what comes next, and we need a way to judge how well it has done.

To calculate how good a set of predictions is, you could multiply the probabilities assigned to the correct characters. However, because each value is between 0 and 1, multiplying them together very quickly results in a tiny number which is hard for a computer to represent and not very nice to work with.

For this reason, it is common to take the [log](https://www.mathsisfun.com/algebra/exponents-logarithms.html) of the probablility, known as the **log likelyhood**. This has two benefits:

- It has a range from -infinity to zero (i.e. log(0) is -inf and log(1) is zero).

- Where you would multiply probabilities, you *add* their logarithms. This prevents the result from getting tiny.

If we take the *negative* of this value we get a range from zero to infinity - the **negative log likelyhood**, a measure of how *bad* our predictions were rather than how good.

Finally, you divide by the count of samples to get the mean - the **average negative log likelyhood**.

If our goal for the model is to minimise this value representing how bad we are, we are effectively saying 'maximise the likelyhood of the predictions being correct' - just what we want!

The output of our model will be a value for each token in our vocabulary. We will interpret the value as the 'log counts' (or [logits](https://deepai.org/machine-learning-glossary-and-terms/logit)) - simply the log of the odds of that token occuring.

Since the reverse of the log function is exponentiation (see the article linked above) we `exp` the logits to get the actual counts for a given token, and then divide that by the sum of all the counts to normalise the value (i.e. now the sum of all probabilities of all tokens will add to 1). This operation is known as the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function. It has the effect of exaggerating large probabilities and minimising small ones.

During training, we are asking the model to guess the next tokens for each sequence in the batch. Of course, we know the answer, so the loss is simply a measure of how confident it was in the correct token vs the others. This is known as [cross entropy](https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression).

> I just found [this](https://www.naukri.com/code360/library/softmax-and-cross-entropy) article which nicely restates much of the above in more detail with examples, as the Wikipedia articles are a bit intense for the uninitiated! Also check out [this Statquest](https://www.youtube.com/watch?v=6ArSys5qHAU&ab_channel=StatQuestwithJoshStarmer). For a closer look at the close relationship between negative log likelyhood and cross entropy loss, [this is a great reference](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81).


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


# Training

## Training loop

## Monitoring

## Experimenting


# Conclusion

## Learning

## Music Generation

## Next Steps