import telebot
from telebot import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Ініціалізація бота
TOKEN = '7069185471:AAFfK6jSHEjYkZIg9ed4_HnfbsF2bjpFULQ'
bot = telebot.TeleBot(TOKEN)

# Завантаження даних фільмів
with open('movies.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
movies = pd.DataFrame(data)

# Перевірка наявності необхідних колонок
required_columns = ['title', 'genres', 'tags', 'rating', 'year', 'url', 'description']
for col in required_columns:
    if col not in movies:
        raise ValueError(f"Файл JSON повинен містити колонку '{col}'.")

# Завантаження даних книг
with open('books.json', 'r', encoding='utf-8') as file:
    book_data = json.load(file)
books = pd.DataFrame(book_data)

required_book_columns = ['title', 'author', 'genres', 'rating', 'year', 'url', 'description']
for col in required_book_columns:
    if col not in books:
        raise ValueError(f"Файл JSON книг повинен містити колонку '{col}'.")

# Завантаження чи створення файлу вподобань
if os.path.exists("user_feedback.json"):
    with open("user_feedback.json", "r", encoding="utf-8") as feedback_file:
        user_feedback = json.load(feedback_file)
else:
    user_feedback = {}


# Підготовка даних
movies['combined_features'] = movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                              movies['tags'].apply(lambda x: ' '.join(x) if x else '')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Змінні для збереження вибору користувача
user_data = {}

# Функція для збереження вподобань
def save_feedback():
    with open("user_feedback.json", "w", encoding="utf-8") as feedback_file:
        json.dump(user_feedback, feedback_file, ensure_ascii=False, indent=4)



# Завантаження даних фільмів
with open('movies.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
movies = pd.DataFrame(data)

# Перевірка наявності необхідних колонок
required_columns = ['title', 'genres', 'tags', 'rating', 'year', 'url', 'description']
for col in required_columns:
    if col not in movies:
        raise ValueError(f"Файл JSON повинен містити колонку '{col}'.")

sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_query_embedding(query):
    """Отримання векторного подання тексту запиту за допомогою Sentence-BERT"""
    return torch.tensor(sentence_model.encode(query))

def get_movie_embeddings(descriptions):
    """Отримання векторного подання описів фільмів за допомогою Sentence-BERT"""
    return [torch.tensor(sentence_model.encode(desc)) for desc in tqdm(descriptions, desc="Обчислення ембеддінгів для описів фільмів")]

# Отримання ембеддінгів для описів фільмів
movies['embedding'] = get_movie_embeddings(movies['description'].tolist())

# Гіпотеза 1: Кількість лайків та рейтинг
def hypothesis_1(like_count, movies):
    movies['score'] = movies['rating'] * 0.7 + like_count * 0.3
    return movies.sort_values(by='score', ascending=False)

# Гіпотеза 2: Історія користувача
def hypothesis_2(user_history, movies):
    if not user_history:
        return movies
    movies['history_score'] = movies['genres'].apply(
        lambda tags: sum(tag in user_history for tag in tags)
    )
    return movies.sort_values(by='history_score', ascending=False)

# Гіпотеза 3: Тренди
def hypothesis_3(movies, trending_now):
    movies['trend_score'] = movies['title'].apply(lambda title: 1 if title in trending_now else 0)
    return movies.sort_values(by='trend_score', ascending=False)

def get_recommendations_with_hypotheses(query, movies, content_type=None, user_history=None, like_count=None, trending_now=None, top_n=5):
    """Пошук рекомендацій з урахуванням гіпотез та типу контенту"""



    query_embedding = get_query_embedding(query)
    movie_embeddings = torch.stack(movies['embedding'].to_list())
    
    # Нормалізація векторів
    query_embedding = query_embedding / query_embedding.norm(dim=0)
    movie_embeddings = movie_embeddings / movie_embeddings.norm(dim=1, keepdim=True)
    
    # Обчислення косинусної подібності
    similarities = torch.mm(movie_embeddings, query_embedding.unsqueeze(1)).squeeze(1)
    movies['similarity'] = similarities
    
    # Фільтрація за типом
    if content_type:
        movies = movies[movies['type'] == content_type]

    # Застосування гіпотез
    if user_history:
        movies = hypothesis_2(user_history, movies)
    if trending_now:
        movies = hypothesis_3(movies, trending_now)

    # Вибір топ-`n` за схожістю
    top_movies = movies.sort_values(by='similarity', ascending=False).head(top_n)

    result = []
    for _, row in top_movies.iterrows():
        result.append({
            f"Назва: {row['title']}\nЖанри: {', '.join(row['genres'])}\nРейтинг: {row['rating']}\n"
            f"Рік: {row['year']}\nКраїна: {', '.join(row['country'])}\n"
            f"Теги: {', '.join(row['tags']) if row['tags'] else 'Немає'}\n"
            f"Посилання: {row['url']}\nОпис: {row['description']}\n"
        })
    return result

    #for _, row in top_movies.iterrows():
    #    result.append({
    #        "Назва": row['title'],
    #        "Тип": row['type'],
    #        "Жанри": ', '.join(row['genres']),
    #        "Рейтинг": row['rating'],
    #        "Рік": row['year'],
    #        "Країна": ', '.join(row['country']),
    #        "Теги": ', '.join(row['tags']) if row['tags'] else "Немає",
    #        "Посилання": row['url'],
    #        "Опис": row['description']
    #    })
    #return result

# Функція для отримання рекомендацій
def get_recommendations_by_user_input(selected_genres, selected_tags, selected_type):
    if selected_type == "book":
        filtered_books = books[books['rating'].notnull() & (books['rating'] != "Рейтинг відсутній")]
        filtered_books = filtered_books[filtered_books['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]

        if filtered_books.empty:
            return "Не знайдено книг, що відповідають вашим критеріям."

        return filtered_books.sort_values(by='rating', ascending=False).reset_index(drop=True)

    # Фільтруємо тільки фільми з рейтингом
    filtered_movies = movies[movies['rating'].notnull() & (movies['rating'] != "Рейтинг відсутній")]

    filtered_movies = filtered_movies[filtered_movies['type'] == selected_type]

    if selected_tags:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(genre in x for genre in selected_genres)) &
                                          filtered_movies['tags'].apply(lambda x: any(tag in x for tag in selected_tags) if x else False)]
    else:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]

    if filtered_movies.empty:
        return "Не знайдено фільмів, що відповідають вашим критеріям."

    filtered_movies = filtered_movies.sort_values(by='rating', ascending=False).reset_index(drop=True)
        # Сортуємо за рейтингом
    return filtered_movies  # Повертаємо до 10 рекомендацій


# Функція для відправки головного меню
def send_main_menu(chat_id):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_movies = types.KeyboardButton("Фільми")
    btn_series = types.KeyboardButton("Серіали")
    btn_anime = types.KeyboardButton("Аніме")
    btn_cartoon = types.KeyboardButton("Мультики")
    btn_books = types.KeyboardButton("Книги")
    markup.add(btn_movies, btn_series,btn_cartoon)
    markup.add( btn_anime, btn_books)
    bot.send_message(chat_id, "Оберіть розділ:", reply_markup=markup)


# Команда /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    send_main_menu(message.chat.id)

@bot.message_handler(func=lambda message: message.text == "Пошук за описом")
def start_search_by_description(message):
    # Вмикаємо режим пошуку
    chat_id = message.chat.id
    message_text = "Приклад для введення: Містична історія про маленького хлопчика\nВведіть опис фільму для пошуку."
    bot.send_message(message.chat.id, message_text)

    # Позначаємо, що користувач у режимі пошуку за описом
    user_data[chat_id].update({'search_mode': 'description'})  # Позначаємо, що користувач у режимі пошуку

def get_liked_count(user_id):
    user_id = str(user_id)  # Переконуємося, що `user_id` — це рядок
    if user_id in user_feedback and 'liked' in user_feedback[user_id]:
        return len(user_feedback[user_id]['liked'])
    return 0
# Функція для завантаження історії користувачів із JSON-файлу
def load_user_history(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            user_feedback = json.load(file)
        user_history = []

        # Об'єднуємо всі вподобані й не вподобані фільми в один список
        for user_id, feedback in user_feedback.items():
            liked_movies = feedback.get('liked', [])
            disliked_movies = feedback.get('disliked', [])

            # Додаємо всі фільми до історії
            user_history.extend([{'user_id': user_id, 'movie': movie, 'feedback': 'liked'} for movie in liked_movies])
            user_history.extend([{'user_id': user_id, 'movie': movie, 'feedback': 'disliked'} for movie in disliked_movies])

        return user_history
    except FileNotFoundError:
        print("Файл не знайдено.")
        return []
    except json.JSONDecodeError:
        print("Помилка в структурі JSON.")
        return []
# Обробка вибору розділу
@bot.message_handler(func=lambda message: message.text in ["Фільми", "Серіали", "Аніме", "Книги","Мультики"])
def handle_category_selection(message):
    user_data[message.chat.id] = {'category': message.text, 'current_recommendation_index': 0}
    # Визначаємо тип на основі вибору користувача
    if message.text == "Фільми":
        selected_type = "film"
    elif message.text == "Серіали":
        selected_type = "serial "
    elif message.text == "Аніме":
        selected_type = "anime"
    elif message.text == "Мультики":
        selected_type = "cartoon"
    else:
        selected_type = 'book'

    user_data[message.chat.id]['type'] = selected_type
    # Кнопки для виводу усіх жанрів або повернення до головного меню
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    #btn_show_genres = types.KeyboardButton("Вивести усі жанри")
    btn_sech_by_descs = types.KeyboardButton("Пошук за жанрами")
    btn_serch_by_genre = types.KeyboardButton("Пошук за описом")
    btn_back = types.KeyboardButton("Головне меню")
    markup.add(btn_back,btn_sech_by_descs,btn_serch_by_genre)
    bot.send_message(message.chat.id, "Оберіть як буде здійсюватися пошук", reply_markup=markup)

# Обробка введеного тексту для пошуку
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('search_mode') == 'description')
def perform_description_search(message):
    search_query = message.text
    user_id = message.chat.id
    chat_id = message.chat.id
    print(user_data)
    index = 0
    #print(user_data['type'])


    if 'search_mode' in user_data[chat_id] and user_data[chat_id]['search_mode'] == 'description':


        print(f"Користувач {message.chat.id} шукає за описом: {search_query}")  # Вивід запиту в консоль
        content_type = user_data[message.chat.id]['type']
        like_count = get_liked_count(user_id)
        user_history = load_user_history("user_feedback.json")
        result = get_recommendations_with_hypotheses (search_query, movies, content_type, user_history, like_count, trending_now=None, top_n=5)
        print(len(result))
        print(result)

        if index < len(result):
            for index in range (0, len(result)):
                result = result[index]
                bot.send_message(chat_id,result)
                print(index)
                index +=1

        # Add buttons for navigation
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn_next = types.KeyboardButton("Наступний фільм")
            btn_like = types.KeyboardButton("Сподобалось")
            btn_dislike  = types.KeyboardButton("Не сподобалось")
            btn_main_menu = types.KeyboardButton("Головне меню")
            markup.add(btn_next, btn_main_menu,btn_like,btn_dislike)
            bot.send_message(chat_id, "Виберіть дію:", reply_markup=markup)
        if index == 5:
            bot.send_message(chat_id, "Це всі доступні рекомендації.")
        # Повертаємо користувача до головного меню після пошуку
        #send_main_menu(message.chat.id)
            send_main_menu(message.chat.id)
            user_data.pop(message.chat.id, None)  # Очищаємо дані про режим пошуку




@bot.message_handler(func=lambda message: message.text == "Пошук за жанрами")
def handle_show_genres(message):
    selected_type = user_data.get(message.chat.id, {}).get('type')
    print(user_data)
    all_genres = set()
    if selected_type == "book":
        for genres in books['genres']:
            all_genres.update(genres)
        print(all_genres)
    else:
        for genres in movies[movies['type'] == selected_type]['genres']:
            all_genres.update(genres)
        print(all_genres)

    all_genres_list = sorted(all_genres)
    max_message_length = 4096
    message_text = "Доступні жанри:\n"
    bot.send_message(message.chat.id, message_text)
    # Розділення на частини
    genre_messages = []
    temp_message = ""
    for genre in all_genres_list:
        if len(temp_message) + len(genre) + 2 <= max_message_length:
            temp_message += genre + "\n"
        else:
            genre_messages.append(temp_message)
            temp_message = genre + "\n"
    genre_messages.append(temp_message)  # Додаємо останню частину

    # Надсилання частин повідомлення
    for part in genre_messages:
        bot.send_message(message.chat.id, part)
    bot.send_message(message.chat.id, "Введіть жанри, які вас цікавлять (напр., 'романтичний, комедія').")


@bot.message_handler(func=lambda message: message.text == "Головне меню")
def handle_main_menu(message):
    send_welcome(message)

# Обробка введення жанрів
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('genres') is None)
def handle_genres(message):
    genres = message.text.split(",")
    user_data[message.chat.id]['genres'] = [genre.strip() for genre in genres]

    # Кнопки для вибору або пропуску тегів
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_skip = types.KeyboardButton("Пропустити")
    btn_back = types.KeyboardButton("Головне меню")
    markup.add(btn_skip, btn_back)
    bot.send_message(message.chat.id,
                     "Введіть хештеги (наприклад, '#детектив'). Якщо хештеги не потрібні, натисніть 'Пропустити'.",
                     reply_markup=markup)


# Обробка тегів або пропуску
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get('tags') is None)
def handle_tags(message):
    if message.text == "Пропустити":
        user_data[message.chat.id]['tags'] = []
    elif message.text == "Головне меню":
        send_main_menu(message.chat.id)
        user_data.pop(message.chat.id, None)  # Очищаємо дані користувача
        return
    else:
        tags = message.text.split(",")
        user_data[message.chat.id]['tags'] = [tag.strip() for tag in tags]

    selected_genres = user_data[message.chat.id].get('genres', [])
    selected_tags = user_data[message.chat.id].get('tags', [])
    selected_type = user_data[message.chat.id].get('type', None)

    recommendations = get_recommendations_by_user_input(selected_genres, selected_tags, selected_type)

    if isinstance(recommendations, str):
        bot.send_message(message.chat.id, recommendations)
        user_data.pop(message.chat.id, None)  # Очищаємо дані користувача
    else:
        user_data[message.chat.id]['recommendations'] = recommendations
        send_recommendations(message.chat.id)


# Function to send recommendations
def send_recommendations(chat_id):
    index = user_data[chat_id]['current_recommendation_index']
    recommendations = user_data[chat_id]['recommendations']
    selected_type = user_data[chat_id]['type']

    if index < len(recommendations):
        row = recommendations.iloc[index]

        if selected_type == 'book':
            # Customized message for books
            bot.send_message(
                chat_id,
                f"Назва: {row['title']}\nАвтор: {row['author']}\nРейтинг: {row['rating']}\n"
                f"Рік: {row['year']}\nЖанри: {', '.join(row['genres'])}\n"
                f"Посилання: {row['url']}\nОпис: {row['description']}\n"
                f"Читачі також насолоджувались:\n" +
                "\n".join([f"{book['title']} by {book['author']} - {book['url']}" for book in row['readers_also_enjoyed']])
            )
        else:
            # Message format for movies/series/anime
            bot.send_message(
                chat_id,
                f"Назва: {row['title']}\nЖанри: {', '.join(row['genres'])}\nРейтинг: {row['rating']}\n"
                f"Рік: {row['year']}\nКраїна: {', '.join(row['country'])}\n"
                f"Теги: {', '.join(row['tags']) if row['tags'] else 'Немає'}\n"
                f"Посилання: {row['url']}\nОпис: {row['description']}\n"
            )

        # Update index for next recommendation
        user_data[chat_id]['current_recommendation_index'] += 1

        # Add buttons for navigation
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn_next = types.KeyboardButton("Наступний фільм")
        btn_like = types.KeyboardButton("Сподобалось")
        btn_dislike  = types.KeyboardButton("Не сподобалось")
        btn_main_menu = types.KeyboardButton("Головне меню")
        markup.add(btn_next, btn_main_menu,btn_like,btn_dislike)
        bot.send_message(chat_id, "Виберіть дію:", reply_markup=markup)
    else:
        bot.send_message(chat_id, "Це всі доступні рекомендації.")



# Обробка натискання кнопки "Наступні 5 фільмів"
@bot.message_handler(func=lambda message: message.text == "Наступний фільм")
def handle_next_recommendations(message):
    send_recommendations(message.chat.id)



# Обробка кнопок "Сподобалось" і "Не сподобалось"
@bot.message_handler(func=lambda message: message.text in ["Сподобалось", "Не сподобалось"])
def handle_feedback(message):
    chat_id = message.chat.id
    index = user_data[chat_id]['current_recommendation_index'] - 1
    recommendations = user_data[chat_id]['recommendations']
    row = recommendations.iloc[index]
    user_id = str(chat_id)
    # Додавання нового користувача, якщо його ще немає в user_feedback
    if user_id not in user_feedback:
        user_feedback[user_id] = {'liked': [], 'disliked': []}
    # Збереження вподобань
    user_id = str(chat_id)
    title = row['title']
    if message.text == "Сподобалось":
        # Видалити фільм з категорії "Не сподобалось", якщо він там є
        if title in user_feedback[user_id]['disliked']:
            user_feedback[user_id]['disliked'].remove(title)
        # Додати фільм у категорію "Сподобалось", якщо його ще немає
        if title not in user_feedback[user_id]['liked']:
            user_feedback[user_id]['liked'].append(title)
    elif message.text == "Не сподобалось":
        # Видалити фільм з категорії "Сподобалось", якщо він там є
        if title in user_feedback[user_id]['liked']:
            user_feedback[user_id]['liked'].remove(title)
        # Додати фільм у категорію "Не сподобалось", якщо його ще немає
        if title not in user_feedback[user_id]['disliked']:
            user_feedback[user_id]['disliked'].append(title)

    save_feedback()  # Зберігаємо зміни у файл


    bot.send_message(chat_id, f"Ваш вибір '{message.text}' збережено.")
    send_recommendations(message.chat.id)


# Запуск бота
bot.polling(none_stop=True)
