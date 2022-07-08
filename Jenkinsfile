
pipeline {

    
    agent any

    stages {

        stage("Download-data-build-test"){

             steps {

                                            sh '''
                                                 docker version
                                                 docker info
                                                 docker build -f Dockerfile.test -t build-image-test .
                                            '''


             }

        }

        stage("build-container-test") {
            

                 
             steps {

                                            

                                                sh '''
                                                 docker run build-image-test
                                                '''
                                            
            
                 }
            }

         
        
        stage("build-image-pypi") {
                 
             steps {

                                                 sh '''
                                                 docker version
                                                 docker info
                                                 docker build -f Dockerfile.publish -t build-image-pypi .
                                                 '''

            
                 }
            }
    
        stage("build-container-pypi") {
            

                 
             steps {

                 withCredentials([
                              usernamePassword(credentialsId: 'twine-login-info',
                              usernameVariable: 'username',
                              passwordVariable: 'password')
                                              ]) 

                                              {

                                                 sh '''
                                                 docker run --env username=${username} --env password=${password} build-image-pypi
                                                 '''
                                              }
            
                 }
            }

    
}

}

