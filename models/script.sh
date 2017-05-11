# testing bugs:
python kullback_ProbPreserv2.py 0 0 10000 3000 500 10;
python kullback_ProbPreserv2.py 1 0 10000 3000 500 10;
python kullback_ProbPreserv2.py 2 0 10000 3000 500 10;
python kullback_ProbPreserv2.py 0 1 10000 3000 500 10;
python kullback_ProbPreserv2.py 1 1 10000 3000 500 10;
python kullback_ProbPreserv2.py 2 1 10000 3000 500 10;
python kullback_ProbPreserv2.py 0 2 10000 3000 500 10;
python kullback_ProbPreserv2.py 1 2 10000 3000 500 10;
python kullback_ProbPreserv2.py 2 2 10000 3000 500 10;
git add -A;
git commit -m "test from ce2";
git pull;
git push;
