![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Music recommendations

## Getting Started

Follow the instructions provided in the notebook.

Read the instructions for each cell and provide your answers. Make sure to test your answers in each cell and save. Jupyter Notebook should automatically save your work progress. But it's a good idea to periodically save your work manually just in case.

## Deliverables

- Downloaded notebook with your responses to each of the exercises.


## Submission

- Upon completion, add your deliverables to git. 
- Then commit git and push your branch to the remote.
- Make a pull request and paste the PR link in the submission field in the Student Portal.

<br>

**Good luck!**

## Completed local solution

Open `lab-music-recommendations.ipynb` and run all cells from this directory:

```sh
python3 -m pip install -r requirements.txt
python3 -m notebook lab-music-recommendations.ipynb
```

The solution includes the MusicCaps metadata, local TF-IDF/cosine retrieval,
preference filters, source-grounded explanations, examples, and executed checks.
The default generator uses templates and consumes no API tokens. An optional
Ollama function provides local LLM generation when a model is installed.
Dataset attribution and license are in `data/README.md`.

The original walkthrough is preserved in `reference/pinecone-audio-demo.ipynb`;
its Pinecone/PANNs workflow was not executed. The optional Ollama path was also
not executed. All default solution code cells were executed and outputs saved.
