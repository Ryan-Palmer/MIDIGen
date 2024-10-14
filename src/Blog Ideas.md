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


# Training

## Training loop

## Monitoring

## Experimenting


# Conclusion

## Learning

## Music Generation

## Next Steps