#include "OutputPathByDate.h"

#include <ctime>



OutputPathByDate::OutputPathByDate() {
	// Set the date when initialized so that it is constant for any function that uses it.
    reset();
}


OutputPathByDate::reset() {
    date = generateOutputPathByDate();
}


std::string OutputPathByDate::generateOutputPathByDate() {
	time_t now;
	const int MAX_DATE = 64;
	char theDate[MAX_DATE];
	theDate[0] = '\0';

	now = time(nullptr);

	if (now != -1) {
		strftime(theDate, MAX_DATE, "/%Y.%m.%d/%H-%M-%S/", gmtime(&now));
		return theDate;
	}
	return "";
}


std::string OutputPathByDate::getPath(std::string seed) {
    return "/" + seed + "/" + date;
}