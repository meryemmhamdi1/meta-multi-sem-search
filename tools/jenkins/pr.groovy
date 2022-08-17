#!/usr/bin/env groovy

@Library('algpipeline') _

// This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
properties([
  buildDiscarder(logRotator(daysToKeepStr: '180', numToKeepStr: '900'))
])

// Please be mindful of the limited resources on Jenkins when setting the timeout and issuing retries for jobs.
// If you need to replay a job on only one platform, please comment the other platforms out for the replay to conserve build resources
def profiles = [
  [host:'mac', name: 'MacOS', label: 'builder&&mac&&!highend', timeout: '10', timeout_unit: 'MINUTES'],
  [host:'win', name: 'Windows' , label: 'builder&&win&&!highend', timeout: '10', timeout_unit: 'MINUTES'],
  [host:'ubuntu', name: 'Ubuntu' ,  label: 'ITC-ubuntu-vm', timeout: '10', timeout_unit: 'MINUTES'],
]

def runInConda(profile, String command, Boolean returnStdOut = false, venv = '.venv') {
  def cmd = ''

  if (profile.host == 'win') {
    cmd = cmd + "& ./.miniconda3/shell/condabin/conda-hook.ps1\n"
    cmd = cmd + 'conda activate '+ venv + '/\n'
    cmd = cmd + '$ENV:PYTHONIOENCODING="utf-8"\n'
    cmd = cmd + command
    echo "Running `${command}`"
    return powershell(
      label: command,
      returnStdout: returnStdOut,
      script: cmd,
    )
  } else {
    cmd = cmd + 'set +x\n'
    cmd = cmd + ". ./.miniconda3/bin/activate\n"
    cmd = cmd + 'conda activate ' + venv + '/\n'
    cmd = cmd + command
    echo "Running `${command}`"
    return sh(
      label: command,
      returnStdout: returnStdOut,
      script: cmd,
    )
  }
}

def build_py_project_scaffold(profile) {
  node(profile.label) {
    timeout([time: profile.timeout, unit: profile.timeout_unit]) {
      withCredentials([
        // These are credentials available on Jenkins. This will make them available as env vars for build.
        sshUserPrivateKey(credentialsId: 'c4279c21-d85b-4493-afdf-2677507825b5', keyFileVariable: 'SSH_KEY_PATH', usernameVariable: 'eucbot'),
        usernamePassword(credentialsId: '4a3cae41-d419-427f-a9b7-2724755a3216', passwordVariable: 'ARTIFACTORY_API_KEY', usernameVariable: 'ARTIFACTORY_USERNAME')
      ]) {

        stage("Setup") {
          printEnvVarsAndJobParams()
          checkout scm
          if (profile.host == 'win') {
            powershell script: "powershell ./tools/install-conda.ps1 ./.miniconda3"
            powershell script: "& ./.miniconda3/shell/condabin/conda-hook.ps1; conda env update --prefix ./.venv --file tools/conda.yaml"
          } else {
            sh script: "bash ./tools/install-conda.sh ./.miniconda3"
            sh script: ". ./.miniconda3/bin/activate; conda env update --prefix ./.venv --file tools/conda.yaml"
          }
          runInConda(profile, "pip install -e .")
          runInConda(profile, "python tools/artifacts.py pull")
        } // Setup

        stage("Lint") {
          def linter_failures = []

          try {
            runInConda(profile, "python ./tools/lint.py pylint")
          } catch (e) {
            linter_failures.add("Running pylint")
          }

          try {
            runInConda(profile, "python ./tools/lint.py mypy")
          } catch (e) {
            linter_failures.add("Running mypy")
          }

          try {
              runInConda(profile, "python ./tools/lint.py isort --check")
          } catch (e) {
              linter_failures.add("Running isort")
          }

          try {
            runInConda(profile, "python ./tools/lint.py black --check")
          } catch (e) {
            linter_failures.add("Running black")
          }

          if (!linter_failures.isEmpty()) {
            String msg = "The following linter phases have failed\n"
            for (String m in linter_failures) {
              msg = msg + m + "\n"
            }
            msg = msg + "To see the faulty files, see the corresponding section of each linter phase.\n"
            error(msg)
          }
        } // Lint
        stage("Test") {
          try {
            runInConda(profile, "python tests/main.py --xml=.tmp_test_out/${profile.name}/test_report.xml --log=.tmp_test_out/${profile.name}/test_log.txt")
          } finally {
            archiveArtifacts artifacts: ".tmp_test_out/${profile.name}/test_log.txt"
            junit testResults: ".tmp_test_out/${profile.name}/test_report.xml"
          }
        } // Test
      } // with credentials
    } // timeout
  } // node
} // Build

// Cancel old jobs on the same branch
def buildNumber = BUILD_NUMBER as int;
if (buildNumber > 1) milestone(buildNumber - 1);
milestone(buildNumber)

timestamps {
  parallel profiles.collectEntries { profile ->
    [ "${profile.name}" : {
      build_py_project_scaffold(profile)
    } ]
  }
}
