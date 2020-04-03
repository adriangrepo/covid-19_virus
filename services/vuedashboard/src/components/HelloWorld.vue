<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
  </div>
</template>

<script>
  import axios from 'axios'

  export default {
    name: 'HelloWorld',
    data() {
      return {
        msg: 'Welcome to Your Vue.js App',
      };
    },
    methods: {
      getMessage() {
        const path = 'http://localhost:8000/';
        axios.get(path)
          .then((res) => {
            this.msg = `Hello, ${res.data.hello}!`;
          })
          .catch((error) => {
            // eslint-disable-next-line
            console.error(error);
          });
      },
    },
    created() {
      this.getMessage();
    },
  };
</script>
