
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
                              passwordVariable: 'password',
                              ),
                              usernamePassword(credentialsId: 'git-login-info-token	',
                              usernameVariable: 'gitusername',
                              passwordVariable: 'gitpassword',
                              )
                              
                                              ]) 

                                              {

                                                 sh '''
                                                 docker run --env username=${username} --env password=${password} --env gitusername=${gitusername}  --env gitpassword=${gitpassword} build-image-pypi
                                                 '''
                                              }
            
                 }
            }

    
}

}

