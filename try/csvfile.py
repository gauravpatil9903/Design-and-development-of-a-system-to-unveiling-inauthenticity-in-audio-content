import csv

# Data for the CSV file
data = [
    ["Mahatma Gandhi", 
     """Mahatma Gandhi, born Mohandas Karamchand Gandhi on October 2, 1869, in Porbandar, India, was the preeminent leader of the Indian independence movement against British colonial rule. He is renowned worldwide for his philosophy of nonviolent resistance, known as 'Satyagraha,' which sought to achieve social and political change without resorting to violence. Gandhi's principles of truth, nonviolence, and civil disobedience became powerful tools in the struggle for freedom and influenced many movements for civil rights and social justice across the globe.

Educated in law in London, Gandhi initially practiced as a lawyer in South Africa, where he first began developing his ideas on passive resistance in response to the racial discrimination he and others faced. In 1915, he returned to India and quickly became a leader in the Indian National Congress, advocating for self-rule (Swaraj) and the upliftment of the poor, particularly the rural peasants.

One of his most famous acts of civil disobedience was the Salt March in 1930, a 240-mile protest against the British monopoly on salt production. This march gained worldwide attention and symbolized India's refusal to be subjugated by unjust laws. Gandhi was also instrumental in campaigns against the oppressive British taxation system, the caste system, and the rights of untouchables, whom he called 'Harijans' or children of God.

Gandhi's efforts culminated in India's independence in 1947. However, the country was partitioned into Hindu-majority India and Muslim-majority Pakistan, a decision that deeply saddened Gandhi, as he had hoped for a united, pluralistic India. He spent his final years advocating for peace between Hindus and Muslims, but on January 30, 1948, he was assassinated by Nathuram Godse, a Hindu nationalist who opposed Gandhi's efforts to reconcile the two communities. Despite his death, Gandhi's legacy lives on, and he is remembered as the Father of the Nation in India and a symbol of peace and nonviolent resistance worldwide.""",
    ],
    ["Mobile Phone",
     """A mobile phone, also known as a cellphone or smartphone, is a portable device that allows users to make calls, send text messages, and access a wide range of applications and services over a cellular network. The mobile phone has revolutionized communication since its inception and has become an integral part of daily life, allowing people to stay connected regardless of location.

The first mobile phones, introduced in the 1970s, were bulky, expensive, and limited to voice communication. Over the decades, advances in technology have led to significant reductions in size, cost, and weight while vastly improving functionality. Modern smartphones are essentially mini-computers, capable of not only voice communication but also internet browsing, social media access, gaming, photography, video recording, navigation, and much more.

Mobile phones operate on a network of cell towers, enabling calls to be routed from one tower to another as the user moves from place to place. Most modern phones also have access to 4G and 5G networks, which offer high-speed internet and support a wide variety of data-intensive applications.

The introduction of touchscreens, mobile apps, and advanced operating systems (like Android and iOS) has transformed mobile phones into powerful tools for both personal and professional use. Apps have turned phones into everything from cameras to GPS devices, fitness trackers, and even virtual assistants. Features such as cameras, high-resolution displays, and large storage capacities have also turned mobile phones into multimedia hubs, allowing users to create, store, and share content in real-time.

Today, mobile phones are ubiquitous, with billions of people around the world relying on them for communication, entertainment, education, and commerce. The advent of 5G technology promises to further revolutionize the mobile experience, enabling faster speeds and supporting the Internet of Things (IoT), autonomous vehicles, and smart cities.""",
    ],
    ["Laptop",
     """A laptop, also known as a notebook computer, is a portable personal computer that integrates all the essential components of a desktop computer into a single, compact unit. The term 'laptop' was coined to describe a device that could be used on one's lap, unlike traditional desktop computers, which require a fixed workstation. Laptops are equipped with a screen, keyboard, touchpad or pointing device, and internal hardware such as a processor, memory, and storage, all housed in a lightweight, foldable design.

Laptops were initially developed in the 1980s as business tools for mobile professionals who needed computing power while traveling. Early models were limited in functionality and battery life, but technological advancements quickly made them more powerful, versatile, and affordable. Today, laptops are widely used for a variety of purposes, including work, education, entertainment, gaming, and creative tasks like video editing, graphic design, and programming.

Modern laptops come in various forms, ranging from ultra-thin and lightweight models designed for portability to high-performance gaming laptops with dedicated graphics cards and powerful processors. Most laptops feature rechargeable batteries, allowing them to operate without being plugged into a power source for several hours, making them ideal for users on the go.

Laptops run on operating systems such as Microsoft Windows, macOS, or Linux, and offer a full range of connectivity options, including Wi-Fi, Bluetooth, and USB ports. They are also equipped with webcams, microphones, and speakers, making them suitable for video conferencing and multimedia consumption.

In recent years, laptops have evolved to include hybrid models, known as 2-in-1 laptops, which can function as both a traditional laptop and a tablet. These devices feature touchscreen displays and can be used with a stylus for drawing or note-taking.

Laptops have become an essential tool for productivity and entertainment in the digital age, offering users the flexibility to work, learn, and create from virtually anywhere.""",
    ]
]

# Filepath for saving the CSV
file_path = "data.csv"

# Writing the data to CSV
with open(file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Information"])
    writer.writerows(data)

file_path
