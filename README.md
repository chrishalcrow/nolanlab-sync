# Nolanlab sync

Package to help synchronize behavioural data to ephys data in the NolanLab. The script `scripts/wolf/synchronise_behaviour.py` takes in bonsai/blender output and an ephys recording and outputs synchronised bonsai/blender output.

Read about the entire NolanLab pipeline: https://github.com/MattNolanLab/analysis_pipelines

This repo represents a _minimum viable product_: it contains a working synchronization pipeline. But it has been forked and modified when applied to other projects in the lab. The modified repos can be found here:

- https://github.com/wulfdewolf/mmnav (private repo; if you're in the lab, please contact Wolf for access)


## Use on your own computer

To begin using this repo, please download (clone) the repo from github and **c**hange **d**irectory into the folder

```
git clone https://github.com/MattNolanLab/nolanlab-sync
cd nolanlab-sync
```

Then you can run anything you'd like using (`uv`)[https://docs.astral.sh/uv/getting-started/installation/] e.g.

```
uv run scripts/template/synchronise_behaviour.py 5 4 MMNAV1
```

Read more about the `synchronise_behaviour.py` script by opening the file: there's lots of documentation inside.

