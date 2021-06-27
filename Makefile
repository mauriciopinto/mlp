CC=gcc
CFLAGS=-g -lm
ALLOBJ=main.o mlp.o
DEPS=mlp.h
ALLX=mlp
OUT=build


.PHONY: all


all:
	make $(ALLX)
	make create_build

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ %< $(CFLAGS) 

mlp: main.o mlp.o
	$(CC) -o $@ $^ $(CFLAGS)

create_build:
	if [ -d $(OUT) ]; then rm -r -f $(OUT); fi
	mkdir $(OUT)
	mv $(ALLOBJ) $(ALLX) $(OUT)


.PHONY: clean


clean:
	rm -r -f $(OUT)





