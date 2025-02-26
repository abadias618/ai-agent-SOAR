# ai-agent-SOAR

# Dataset
The {Jericho Dataset https://github.com/JerichoWorld/JerichoWorld}
### EDA
There seems to be 27 "rom"s, meaning 27 games they come in train.json for each
rom there are a list of iterations. So, when you read the json you basically encounter this:
[[{rom:"x",...},{rom:"x",...}],
[{rom:"y",...},{rom:"y",...}]]