#pragma gaussian

#include <vector>
#include <string>

class ImageConvolver {
public:
    /**
     * @brief Конструктор принимает ядро свертки и его размеры.
     * Ядро сохраняется внутри класса для последующего использования.
     */
    ImageConvolver(const std::vector<float>& kernel, int kW, int kH);

    /**
     * @brief Загружает изображение с диска.
     * 
     * @param filename Путь к файлу.
     * @param w [out] Сюда запишется ширина.
     * @param h [out] Сюда запишется высота.
     * @param channels [out] Сюда запишется количество каналов.
     * @return unsigned char* Указатель на буфер изображения (выделен через malloc в stbi).
     *         Пользователь обязан освободить его через stbi_image_free().
     */
    unsigned char* loadImage(const char* filename, int& w, int& h, int& channels);

    /**
     * @brief Выполняет свертку RGB изображения.
     * Картинка передается по указателю, результат возвращается вектором (RAII).
     * 
     * @param img_in Указатель на исходные данные.
     * @param w Ширина изображения.
     * @param h Высота изображения.
     * @return std::vector<unsigned char> Буфер с обработанным изображением.
     */
    std::vector<unsigned char> process_default(const unsigned char* img_in, int w, int h);
    std::vector<unsigned char> process_SIMD(const unsigned char* img_in, int w, int h);

    /**
     * @brief Сохраняет изображение на диск (в формате JPG).
     * 
     * @param filename Путь для сохранения.
     * @param w Ширина.
     * @param h Высота.
     * @param data Указатель на данные изображения.
     * @return true если успешно, false если ошибка.
     */
    bool saveImage(const char* filename, int w, int h, const unsigned char* data);

private:
    // Внутреннее состояние: только параметры ядра
    std::vector<float> m_kernel;
    int m_kW;
    int m_kH;
};
