CC=g++
CFLAGS=-g -lm
ALLOBJ=main.o mlp.o parser.o
DEPS=mlp.h parser.h
ALLX=mlp
OUT=build


.PHONY: all


all:
	make $(ALLX)
	make create_build

%.o: %.cxx $(DEPS)
	$(CC) -c -o $@ %< $(CFLAGS)

mlp: main.o mlp.o parser.o
	$(CC) -o $@ $^ $(CFLAGS)

create_build:
	if [ -d $(OUT) ]; then rm -r -f $(OUT); fi
	mkdir $(OUT)
	mv $(ALLOBJ) $(ALLX) $(OUT)


.PHONY: clean


clean:
	rm -r -f $(OUT)





