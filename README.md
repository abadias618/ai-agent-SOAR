# ai-agent-SOAR

# Dataset
Using the Jericho repo for AI agents form Microsoft
### EDA
- Getting the games from [jericho-game-suit https://github.com/BYU-PCCL/z-machine-games/tree/master/jericho-game-suite]
- Did a manual download of the rom from Github, but could also wget the whole suit of games as instructed in the official Jericho docs

# SOAR Architecture
- Here I'm using the extended SOAR architecture proposed in 2007 (gotta check the date better) SOAR 9 I think.

# TODOs
- ~~Implement Episodic Memory.~~ (Mar 25, 02:16am)
- ~~Implement Long and Short term memory for images (Visual Memory).(Apr 1, 02:42am)~~
- Implement Apraisal Detector system.
- Implement Decision Procedure.
- ~~Restructure how vectors are stored (use FAISS and separate sem, proc, epi, into meta-tags instead of instantiating 3 separate vector DBs)~~
- ~~Figure out how to pass everything to shorterm memory without memory leaking.~~
- Retrieve relevant memories to ST Mem.
