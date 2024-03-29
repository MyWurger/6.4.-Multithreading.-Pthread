#include <iostream>                 // заголовочный файл со стандартной библиотекой ввода/вывода
#include <opencv2/opencv.hpp>       // заголовок, подтягивающий все функции OpenCV
#include <chrono>                   // заголовочный файл для работы с временем и измерениями времени выполнения задач
#include <pthread.h>                // заголовочный файл, предназначенный для работы с потоками в многопоточном программировании

using namespace std;                // используем пространство имён std
using namespace cv;                 // используем пространство имён cv



/**************************************************************************/
/*   Г Л О Б А Л Ь Н Ы Е    К О Н С Т А Н Т Ы   И   П Е Р Е М Е Н Н Ы Е   */
/**************************************************************************/

const float sensitivityFactor = 1;  // коэффициент чувствительности для масштабирования значения градиентов и увеличения их чувствительности

// структура для работы с фильтром Собеля в потоках pthread.h
struct ThreadData 
{
    Mat inputImage;                 // входное изображение
    Mat outputImage;                // выходное изображение для заданного потока
    int startRow;                   // начало диапазона строк, рассматриваемых потоком
    int endRow;                     // конец диапазона строк, рассматриваемых потоком
};// ThreadData



/**************************************************************************/
/*                  П Р О Т О Т И П Ы   Ф У Н К Ц И Й                     */
/**************************************************************************/

// функция, выполняющая операцию Sobel на изображении в YUV цветовом пространстве в указанном диапазоне строк
void sobelYUVWithRange(const Mat& inputImage,  // входное изображение в формате const Mat, чтобы не поменять
                       Mat& outputImage,       // выходное изображение в формате Mat куда будет записан результат потока
					   int startRow,           // начальная строка, с которой будет производиться операция Sobel
					   int endRow);            // конечная строка, до которой будет производиться операция Sobel

// функция, которая будет использоваться в качестве точки входа для выполнения операции Sobel в отдельном потоке
void* SobelThread(void* threadData);           // данные, передаваемые для выполнения операции Sobel в потоке

// печать приветственной надписи на экран
void print_start();

// печать прощальной надписи на экран
void print_end();



/*------------------------------------------------------------------------*/
/*                Функции               */
/*--------------------------------------*/

// функция, которая будет использоваться в качестве точки входа для выполнения операции Sobel в отдельном потоке
void* SobelThread(void* threadData)            // данные, передаваемые для выполнения операции Sobel в потоке. Должны быть void*        
{
	// приводим указатель на данные к типу ThreadData
    ThreadData* data = static_cast<ThreadData*>(threadData);

	// вызываем функцию sobelYUVWithRange, передавая необходимые данные для операции Sobel
    sobelYUVWithRange(data->inputImage, data->outputImage, data->startRow, data->endRow);

	// завершаем поток после отработки
	pthread_exit(NULL);
}



// функция, выполняющая операцию Sobel на изображении в YUV цветовом пространстве в указанном диапазоне строк
void sobelYUVWithRange(const Mat& inputImage,  // входное изображение в формате const Mat, чтобы не поменять
                       Mat& outputImage,       // выходное изображение в формате Mat куда будет записан результат потока
					   int startRow,           // начальная строка, с которой будет производиться операция Sobel
					   int endRow)             // конечная строка, до которой будет производиться операция Sobel
{
    Mat yuvImage;                              // матрица для копирования изображения в YUV формате

	// перевод входного изображения в формат YUV
	// аргументы: входное изображение в формате RGB, выходное изображение и флаг COLOR_BGR2YUV, указывающий на тип преобразования. В данном случае, используется преобразование из BGR в YUV 
    cvtColor(inputImage, yuvImage, COLOR_BGR2YUV);

    Mat channels[3];                           // создается массив матриц для разделения YUV на отдельные каналы

	// разделение изображение YUV на отдельные каналы и сохранение каждого канала в соответствующем элементе массива
	// Y-канал в channels[0]
	// U-канал в channels[1]
	// V-канал в channels[2]
    split(yuvImage, channels);
   
	// работаем в пространстве яркости
    Mat yChannel = channels[0];                // сохраняем содержимое о яркости изображения в отдельный канал
    Mat gradientX, gradientY;                  // матрицы для хранения горизонтального и вертикального градиентов соответственно

    // убеждаемся, что startRow и endRow находятся в пределах изображения
	// проверяется, чтобы startRow было не меньше 0, и endRow не превышало количество строк в матрице yChannel
    startRow = max(0, startRow); 
    endRow = min(yChannel.rows - 1, endRow);
    
	// проход по строкам изображения от 'startRow' до 'endRow'
    for (int y = startRow; y <= endRow; y++)
    {
		// сохранены результаты оператора Собеля для горизонтального и вертикального градиента соответственно
        Mat gradientXRow, gradientYRow;

		// применяем фильтр Собеля
		// yChannel.row(y) - это конкретная строка изображения, к которой применяется оператор Собеля для вычисления градиентов по горизонтали (gradientXRow) и по вертикали (gradientYRow)
		// CV_32F - тип данных для хранения градиентов, в данном случае - 32-битное числовое представление с плавающей точкой
		// 1, 0 - параметры, указывающие порядок производных по x и y соответственно.
		// в первом случае вычисляется градиент по x - горизонтальный градиент, а во втором - по y - вертикальный градиент
		// 3 - размер ядра оператора Собеля, в данном случае 3х3 - используемые матрицы - влияет на области градиентов
		// 3 - масштабный коэффициент - отвечает за масштабирование результата вычисления градиента. Этот параметр позволяет управлять чувствительностью оператора Собеля к изменениям яркости пикселей в изображении. Установка данного параметра на 1 обычно означает использование стандартного масштаба для вычисления градиента
        // 0 - смещение оператора. В данном случае смещения нет
		// BORDER_DEFAULT - тип обработки границ изображения. В данном случае используется значение по умолчанию, которое означает отражение пикселей на границах для вычисления градиента на краях изображения
		Sobel(yChannel.row(y), gradientXRow, CV_32F, 1, 0, 3, 3, 0, BORDER_DEFAULT);
        Sobel(yChannel.row(y), gradientYRow, CV_32F, 0, 1, 3, 3, 0, BORDER_DEFAULT);

		// полученные строки добавляются в итоговые матрицы
        gradientX.push_back(gradientXRow * sensitivityFactor);
        gradientY.push_back(gradientYRow * sensitivityFactor);
    }// for y

	// переменная, в которую будет сохранен результат вычисления общей магнитуды градиента - абсолютное значение градиента яркости в каждой точке изображени
    Mat gradientMagnitude;

	// применяет формулу для вычисления общей магнитуды градиента по следующей формуле:
	// gradientMagnitude = sqrt{(gradientX)^2 + (gradientY)^2}
	// этот подход позволяет объединить информацию о градиенте по горизонтали и вертикали в одно значение, которое отображает общую силу изменения яркости в каждой точке изображения
    magnitude(gradientX, gradientY, gradientMagnitude);

	// нормализация значения градиента в заданном диапазоне
	// параметр gradientMagnitude - массив или изображение, значения которого требуется нормализовать
	// параметр gradientMagnitude - массив, в который будут сохранены нормализованные значения
	// параметр 0 - минимальное значение, к которому нужно нормализовать массив
	// параметр 255 - максимальное значение, к которому нужно нормализовать массив
	// параметр NORM_MINMAX - указывает на то, что значения должны быть нормализованы в диапазоне между минимальным и максимальным значениями
    normalize(gradientMagnitude, gradientMagnitude, 0, 255, NORM_MINMAX);

	// преобразования значений в массиве gradientMagnitude в формат с использованием 8-бит беззнаковых целых чисел (CV_8U) и сохранения результата в массиве outputImage
	// значения в массиве gradientMagnitude будут преобразованы в диапазон от 0 до 255, так как тип данных CV_8U представляет значения от 0 до 255
    // любые значения, которые выходят за этот диапазон, будут отсечены или округлены таким образом, чтобы они лежали в пределах от 0 до 255
	gradientMagnitude.convertTo(outputImage, CV_8U);

	// хранение результата гистограмного выравнивания
	Mat equalizedImage;

	// улучшает контраст изображения, применяя гистограммное выравнивание. 
	// Гистограммное выравнивание изменяет распределение яркости пикселей таким образом, чтобы расширить
	// диапазон яркости и улучшить контраст изображения. Этот процесс поможет сделать изображение более четким
	// и выразительным
    equalizeHist(outputImage, equalizedImage);
	equalizedImage.copyTo(outputImage);        // копируем полученное контрастное изображение в результат

    Mat blurred;                               // матрица для хранения размытого изображения

	// применения гауссовского размытия к изображению outputImage
	// параметр Size(0, 0) указывает на размер ядра фильтра размытия (в данном случае, выбирается автоматически в зависимости от значения)
    GaussianBlur(outputImage, blurred, Size(0, 0), 5);

	// смешивание изображений: исходного и размытого. Складывает два изображения с различными коэффициентами и сохраняет результат в третьем изображении (в данном случае, снова сохраняется в outputImage)
	// изображение (outputImage) умножается на коэффициент 1.5
	// второе изображение (blurred) умножается на коэффициент -0.5 (вычитание), и затем результаты суммируются
    addWeighted(outputImage, 1.5, blurred, -0.5, 0, outputImage);
	return;                                    // возвращаем обещанное функцией значение
}



// печать приветственной надписи на экран
void print_start()
{
    // выводим ASCII-арт начала
    printf("\t\033[38;5;219m      ,--.--------.   ,--.-,,-,--,                    ,----.     ,---.                       ,-,--.     \n");
    printf("\t     /==/,  -   , -\\ /==/  /|=|  |   .-.,.---.     ,-.--` , \\  .--.'  \\       _,..---._    ,-.'-  _\\   \n");
    printf("\t     \\==\\.-.  - ,-./ |==|_ ||=|, |  /==/  `   \\   |==|-  _.-`  \\==\\-/\\ \\    /==/,   -  \\  /==/_ ,_.'  \n");
    printf("\t      `--`\\==\\- \\    |==| ,|/=| _| |==|-, .=., |  |==|   `.-.  /==/-|_\\ |   |==|   _   _\\ \\==\\  \\    \n");
    printf("\t           \\==\\_ \\   |==|- `-' _ | |==|   '='  / /==/_ ,    /  \\==\\,   - \\  |==|  .=.   |  \\==\\ -\\     \n");
    printf("\t           |==|- |   |==|  _     | |==|- ,   .'  |==|    .-'   /==/ -   ,|  |==|,|   | -|  _\\==\\ ,\\    \n");
    printf("\t           |==|, |   |==|   .-. ,\\ |==|_  . ,'.  |==|_  ,`-._ /==/-  /\\ - \\ |==|  '='   / /==/\\/ _ |   \n");
    printf("\t           /==/ -/   /==/, //=/  | /==/  /\\ ,  ) /==/ ,     / \\==\\ _.\\=\\.-' |==|-,   _`/  \\==\\ - , /  \n");
    printf("\t           `--`--`   `--`-' `-`--` `--`-`--`--'  `--`-----``   `--`         `-.`.____.'    `--`---'\033[0m \n\n");
    return;                                    // вернули обещанное функцией значение             
}



// печать прощальной надписи на экран
void print_end()
{
    // выводим ASCII-арт начала
    printf("\n\t\t\033[38;5;86m ,--.--------.   ,--.-,,-,--,      ,----.             ,----.   .-._\n");
    printf("\t\t /==/,  -   , -\\ /==/  /|=|  |   ,-.--` , \\         ,-.--` , \\ /==/ \\  .-._    _,..---._\n");
    printf("\t\t \\==\\.-.  - ,-./ |==|_ ||=|, |  |==|-  _.-`        |==|-  _.-` |==|, \\/ /, / /==/,   -  \\ \n");
    printf("\t\t  `--`\\==\\- \\    |==| ,|/=| _|  |==|   `.-.        |==|   `.-. |==|-  \\|  |  |==|   _   _\\ \n");
    printf("\t\t       \\==\\_ \\   |==|- `-' _ | /==/_ ,    /       /==/_ ,    / |==| ,  | -|  |==|  .=.   |\n");
    printf("\t\t       |==|- |   |==|  _     | |==|    .-'        |==|    .-'  |==| -   _ |  |==|,|   | -|\n");
    printf("\t\t       |==|, |   |==|   .-. ,\\ |==|_  ,`-._       |==|_  ,`-._ |==|  /\\ , |  |==|  '='   /\n");
    printf("\t\t       /==/ -/   /==/, //=/  | /==/ ,     /       /==/ ,     / /==/, | |- |  |==|-,   _`/\n");
    printf("\t\t       `--`--`   `--`-' `-`--` `--`-----``        `--`-----``  `--`./  `--`  `-.`.____.'\033[0m\n\n");
    return;                                    // вернули обещанное функцией значение             
}



/**************************************************************/
/*            О С Н О В Н А Я   П Р О Г Р А М М А             */
/**************************************************************/
int main(int argc, char** argv)
{
	system("clear");                           // очистка консоли перед началом работы программы

	// работаем с аргументами консоли
	// должно быть два аргумента: исполняемый файл и путь до обрабатываемой фотографии
	if (argc < 2)
    {
		// вывод сообщения об ошибке при недостаточном количестве аргументов командной строки
        cout << "\033[35m ОШИБКА! Используйте: " << argv[0] << " <path_to_image>. Код ошибки -1\033[0m" << endl;
        return -1;                             // выходим на перезапуск программы с ошибкой
    }// if

	Mat inputImage = imread(argv[1], -1);      // загрузка изображения из указанного пути без изменений, т.к. -1

	// загруженное изображение пусто
	if (inputImage.empty())
    {
		// вывод сообщения об ошибке при некорректной загрузке изображения
        cout << "\033[35m ОШИБКА! Загрузка фото происходит некорректно. Код ошибки -2\033[0m" << endl;
        return -2;                             // выходим на перезапуск программы с ошибкой
    }// if

	// создается новая матрица выходного изображения того же размера, что и inputImage, с одним каналом цвета
	// и типом данных CV_8UC1. Тип CV_8UC1 означает, что каждый пиксель представлен в виде 8-битного
	// беззнакового целого числа (unsigned char), что соответствует черно-белому изображению, т.к. результат будет ЧБ
	Mat outputImage(inputImage.size(), CV_8UC1);

	// массив количества потоков. Значения повторяются, чтобы понять, насколько на время выполнения играют запуски на уже "разогретом процессоре" для данного количества потоков
	int numThreads[14] = {1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64};
	print_start();                             // выводим приветственную надпись

	// проходим по массиву элемента потоков в numThread - очередное количество потоков
	for (int numThread : numThreads)
    {
		pthread_t threads[numThread];          // массив дочерних потоков для основного
		ThreadData data[numThread];            // переменная структуры для хранения данных каждого потока
		
		// определение количества строк исходного изображения на каждый поток. Таким образом гарантируется, что разными потоками не будет доступа к одной
		// и той же области памяти и аргументируется нецелесообразность использования мьютексов для ожидания завершения
		// работы одного потока перед переходом к работе второго, так как будет работать так же, как с одним потоком без параллелизма
		int rowsPerThread = inputImage.rows / numThread;
		// запускаем отсчёт таймера для выполнения
		auto start = chrono::high_resolution_clock::now();
		
		// проходим по всем потокам, которые нам надо создать
		for (int i = 0; i < numThread; ++i)
        {
			int res;                           // переменная - вернувшееся значение после создания нового потока

			// передача данных в структуру для каждого потока
            data[i].inputImage = inputImage;
            data[i].outputImage = outputImage;
			// определение начальной строки для текущего потока
			data[i].startRow = i * rowsPerThread;
			// определение конечной строки для текущего потока
			// если текущий поток является последним (когда i == numThread - 1), то конечная строка устанавливается равной последней строке изображения
			// в противном случае (когда текущий поток не последний), конечная строка вычисляется как начальная строка текущего потока плюс количество строк, обрабатываемых каждым потоком минус одна
			data[i].endRow = (i == numThread - 1) ? inputImage.rows - 1 : data[i].startRow + rowsPerThread - 1;
			
			// создание потока threads[i] и запуск функции void* SobelThread(void* threadData) с передачей данных
			res = pthread_create(&threads[i], NULL, SobelThread, &data[i]);

			// ДАЛЬНЕЙШИЙ КОД РЕАЛИЗУЕТСЯ ТОЛЬКО ОСНОВЫНМ ПОТОКОМ
			// если pthread_create вернул 0, то новый поток создан успешно
			// иначе - ошибка создания нового потока
			if(res != 0)
			{
				cout << "\033[35m ОШИБКА! Поток создан некорректно. Код ошибки -3\033[0m" << endl;
				return -3;                      // вернули обещанное значение - завершили программу
			}// if

		}// for i
		
		// все дочерние потоки созданы
		// ожидание завершения основным потоком работы всех дочерних потоков
		for (int i = 0; i < numThread; ++i) 
		{
			int res;                            // переменная - вернувшееся значение после join
			void*thread_result;                 // указатель на возвращаемое потоком значение

			// ожидание завершения конкретного потока и получение результата
            res = pthread_join(threads[i], &thread_result);

			// завершение join - некорректное
			if(res != 0)
			{
				cout << "\033[35m ОШИБКА! Неудачный join с дочерним потоком. Код ошибки -4\033[0m" << endl;
				return -4;                       // возвращаем обещанное значение - выходим с ошибкой
			}// if
        }// for i

		// захват времени окончания выполнения программы Фильтра собеля для заданного количества потоков
		auto end = chrono::high_resolution_clock::now();
		// вычисление продолжительности выполнения операции для заданного количества потоков
		chrono::duration<double> duration = end - start;
		// выводим количество потоков и затраченное время на выволнение с таким количеством программы
        cout << "Количество потоков: \033[38;5;150m" << numThread << "\033[0m. Длительность обработки: \033[38;5;205m" << duration.count() << "\033[0m секунд." << endl;
	}// for

	// после работы со всеми количествами потоков в outputImage осталось изображение после применения фильтра Собеля для 64 потоков
	// выведем его на экран для красоты
	namedWindow("Original Image", WINDOW_NORMAL);// создание окна исходного изображения с возможностью ручного изменения размера
	namedWindow("Output Image", WINDOW_NORMAL);  // создание окна результирующего изображения с возможностью ручного изменения размера
	resizeWindow("Original Image", 800, 600);    // установка начального размера окна для исходного изображения
	resizeWindow("Output Image", 800, 600);      // установка начального размера окна для результирующего изображения
	imshow("Original Image", inputImage);        // открываем исходное изображение в соответствующем окне
	imshow("Output Image", outputImage);         // открываем результирующее изображение в соответствующем окне
	print_end();	                             // выводим надпись завершения работы
	waitKey(0);                                  // ожидаем нажатия любой клавиши клавиатуры для закрытия созданных ранее окон
	destroyAllWindows();                         // уничтожаем все созданные ранее окна

	return 0;                                    // возвращаем обещанное ранее значение в случае успеха работы программы
}



// Корректный запуск

// dmitru@astralinux:~/Проекты VisualCode/Sobel_Filter$ cd build
// dmitru@astralinux:~/Проекты VisualCode/Sobel_Filter/build$ cmake ..
// -- Configuring done
// -- Generating done
// -- Build files have been written to: /home/dmitru/Проекты VisualCode/Sobel_Filter/build
// dmitru@astralinux:~/Проекты VisualCode/Sobel_Filter/build$ make
// [ 50%] Linking CXX executable Sobel_Filter
// [100%] Built target Sobel_Filter
// dmitru@astralinux:~/Проекты VisualCode/Sobel_Filter/build$ ./Sobel_Filter
// Gtk-Message: Failed to load module "gail"