all:
	cd SRC; make all
	mv SRC/main.e ./RUN
clean:
	cd SRC; make clean
	cd RUN; rm *.e
