import { Title, Container, Main } from '../../components'
import styles from './styles.module.css'
import MetaTags from 'react-meta-tags'

const About = ({ updateOrders, orders }) => {
  
  return <Main>
    <MetaTags>
      <title>О проекте</title>
      <meta name="description" content="Фудграм - О проекте" />
      <meta property="og:title" content="О проекте" />
    </MetaTags>
    
    <Container>
      <h1 className={styles.title}>Привет!</h1>
      <div className={styles.content}>
        <div>
          <h2 className={styles.subtitle}>Что это за сайт?</h2>
          <div className={styles.text}>
            <p className={styles.textItem}>
              Основная идея платформы — создать удобное пространство для любителей кулинарии, где каждый сможет не только сохранять свои фирменные рецепты, но и находить вдохновение в блюдах других пользователей. 
            </p>
            <p className={styles.textItem}>
              Сайт позволяет составлять список покупок для выбранных рецептов, отмечать любимые блюда, а также просматривать рецепты по категории подписок, что делает поиск подходящих блюд ещё более персонализированным.
            </p>
            <p className={styles.textItem}>
              Для доступа ко всем функциям необходимо пройти простую регистрацию. Обратите внимание, что подтверждение email не требуется, поэтому вы можете указать любой адрес для создания аккаунта.
            </p>
            <p className={styles.textItem}>
              Присоединяйтесь к нашему сообществу гурманов, публикуйте свои кулинарные изыски и наслаждайтесь блюдами от других пользователей!
            </p>
          </div>
        </div>
        <aside>
          {/* <h2 className={styles.additionalTitle}>
            
          </h2> */}
          {/* <div className={styles.text}>
            <p className={styles.textItem}>
            </p>
            <p className={styles.textItem}>
            </p>
          </div> */}
        </aside>
      </div>
      
    </Container>
  </Main>
}

export default About

