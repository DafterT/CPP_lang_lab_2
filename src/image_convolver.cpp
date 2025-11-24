#include "image_convolver.h"
#include <iostream>
#include <algorithm>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"
#include <immintrin.h>

ImageConvolver::ImageConvolver(const std::vector<float>& kernel, int kW, int kH)
    : m_kernel(kernel), m_kW(kW), m_kH(kH) 
{
}

unsigned char* ImageConvolver::loadImage(const char* filename, int& w, int& h, int& channels) {
    unsigned char* img = stbi_load(filename, &w, &h, &channels, 4);
    if (!img) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return nullptr;
    }
    channels = 4;
    return img;
}

std::vector<unsigned char> ImageConvolver::process_default(const unsigned char* img_in, int w, int h) {
    if (!img_in) return {};

    std::vector<unsigned char> img_out(w * h * 4);
    
    int kHalfW = m_kW / 2;
    int kHalfH = m_kH / 2;

    // 1. Основная область свертки
    for (int y = kHalfH; y < h - kHalfH; ++y) {
        for (int x = kHalfW; x < w - kHalfW; ++x) {

            float sumR = 0.f, sumG = 0.f, sumB = 0.f;

            for (int ky = -kHalfH; ky <= kHalfH; ++ky) {
                for (int kx = -kHalfW; kx <= kHalfW; ++kx) {
                    int sx = x + kx;
                    int sy = y + ky;
                    int srcIdx = (sy * w + sx) * 4;

                    int kkx = kx + kHalfW;
                    int kky = ky + kHalfH;
                    // Используем m_kernel
                    float wgt = m_kernel[kky * m_kW + kkx];

                    sumR += wgt * img_in[srcIdx + 0];
                    sumG += wgt * img_in[srcIdx + 1];
                    sumB += wgt * img_in[srcIdx + 2];
                }
            }

            int dstIdx = (y * w + x) * 4;
            img_out[dstIdx + 0] = (unsigned char)std::clamp(sumR, 0.f, 255.f);
            img_out[dstIdx + 1] = (unsigned char)std::clamp(sumG, 0.f, 255.f);
            img_out[dstIdx + 2] = (unsigned char)std::clamp(sumB, 0.f, 255.f);
            img_out[dstIdx + 3] = img_in[dstIdx + 3]; 
        }
    }

    // 2. Обработка границ
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y < kHalfH || y >= h - kHalfH ||
                x < kHalfW || x >= w - kHalfW) {
                int idx = (y * w + x) * 4;
                img_out[idx + 0] = img_in[idx + 0];
                img_out[idx + 1] = img_in[idx + 1];
                img_out[idx + 2] = img_in[idx + 2];
                img_out[idx + 3] = img_in[idx + 3];
            }
        }
    }

    return img_out;
}

std::vector<unsigned char> ImageConvolver::process_SIMD(const unsigned char* img_in, int w, int h) {
    if (!img_in) return {};

    std::vector<unsigned char> img_out(w * h * 4);
    
    int kHalfW = m_kW / 2;
    int kHalfH = m_kH / 2;

    int xEnd = w - kHalfW; 
    // Выравниваем цикл по 4 пикселя
    int xSimdEnd = kHalfW + ((xEnd - kHalfW) / 4) * 4;

    for (int y = kHalfH; y < h - kHalfH; ++y) {
        
        int x = kHalfW;
        
        // 4 пикселя за итерацию)
        for (; x < xSimdEnd; x += 4) {
            
            // Аккумулятор на 16 float чисел (4 пикселя * 4 канала)
            // [R1 G1 B1 A1 | R2 G2 B2 A2 | R3 G3 B3 A3 | R4 G4 B4 A4]
            __m512 vSum = _mm512_setzero_ps();

            for (int ky = -kHalfH; ky <= kHalfH; ++ky) {
                for (int kx = -kHalfW; kx <= kHalfW; ++kx) {
                    
                    // 1. Загружаем вес ядра
                    float wgt = m_kernel[(ky + kHalfH) * m_kW + (kx + kHalfW)];
                    __m512 vWgt = _mm512_set1_ps(wgt); // Размножаем вес на все 16 позиций 512-битного регистра

                    // 2. Загружаем 4 пикселя данных
                    int srcIdx = ((y + ky) * w + (x + kx)) * 4;
                    __m128i vPx8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&img_in[srcIdx]));

                    // 3. Разворачиваем 8-битные uchar сразу в 32-битные int (в регистр 512 бит)
                    __m512i vPx32 = _mm512_cvtepu8_epi32(vPx8);

                    // 4. Конвертируем int -> float
                    __m512 vPxFloat = _mm512_cvtepi32_ps(vPx32);

                    // 5. FMA: Sum += Px * Wgt
                    vSum = _mm512_fmadd_ps(vPxFloat, vWgt, vSum);
                }
            }

            // 1. Конвертируем float -> int32
            __m512i vRes32 = _mm512_cvtps_epi32(vSum);

            // 2. Упаковка 32-бит int -> 8-бит uchar
            // clamp (0-255) и упаковку из 512 бит обратно в 128 бит.
            __m128i vRes8 = _mm512_cvtusepi32_epi8(vRes32);

            // 3. Сохраняем 16 байт (4 пикселя) в память
            int dstIdx = (y * w + x) * 4;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&img_out[dstIdx]), vRes8);

            // 4. Восстанавливаем Alpha-канал для 4 пикселей
            img_out[dstIdx + 3]  = img_in[dstIdx + 3];
            img_out[dstIdx + 7]  = img_in[dstIdx + 7];
            img_out[dstIdx + 11] = img_in[dstIdx + 11];
            img_out[dstIdx + 15] = img_in[dstIdx + 15];
        }

        // --- Хвост (дорабатываем оставшиеся)
        for (; x < xEnd; ++x) {
            float sumR = 0.f, sumG = 0.f, sumB = 0.f;
            for (int ky = -kHalfH; ky <= kHalfH; ++ky) {
                for (int kx = -kHalfW; kx <= kHalfW; ++kx) {
                    int srcIdx = ((y + ky) * w + (x + kx)) * 4;
                    float wgt = m_kernel[(ky + kHalfH) * m_kW + (kx + kHalfW)];
                    sumR += wgt * img_in[srcIdx + 0];
                    sumG += wgt * img_in[srcIdx + 1];
                    sumB += wgt * img_in[srcIdx + 2];
                }
            }
            int dstIdx = (y * w + x) * 4;
            img_out[dstIdx + 0] = (unsigned char)std::clamp(sumR, 0.f, 255.f);
            img_out[dstIdx + 1] = (unsigned char)std::clamp(sumG, 0.f, 255.f);
            img_out[dstIdx + 2] = (unsigned char)std::clamp(sumB, 0.f, 255.f);
            img_out[dstIdx + 3] = img_in[dstIdx + 3];
        }
    }
    
    // Очистка верхних битов
    _mm256_zeroupper(); 

    // Обработка границ (копирование)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y < kHalfH || y >= h - kHalfH || x < kHalfW || x >= w - kHalfW) {
                int idx = (y * w + x) * 4;
                img_out[idx + 0] = img_in[idx + 0];
                img_out[idx + 1] = img_in[idx + 1];
                img_out[idx + 2] = img_in[idx + 2];
                img_out[idx + 3] = img_in[idx + 3];
            }
        }
    }

    return img_out;
}

bool ImageConvolver::saveImage(const char* filename, int w, int h, const unsigned char* data) {
    if (!data) return false;
    // Качество JPG 90
    return stbi_write_jpg(filename, w, h, 4, data, 90) != 0;
}
